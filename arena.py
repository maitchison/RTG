"""
Fight agents against each-other, and against scripted responses
"""


# on cpu we take 2 hours to process every checkpoint in rescue, this is fine as we generate a checkpoint every 3-hours
# or so, and I think it'll be more like 30min on GPU. This does mean I'll need to keep the evaluation running as we go though

DEVICE = "cuda:0"
TRIALS = 128

from typing import List, Union
import strategies
from support import make_env, make_algo, Config

import itertools

import rescue
import pickle
import time
import math

import ast
import os
import torch
import utils
import matplotlib.pyplot as plt

import numpy as np

class BaseController():
    """
    Controller for game, plays one side only, must maintain it's rnn states
    """

    def __init__(self, name="controller"):
        self.players = None
        self.name = name

    def forward(self, observations):
        """
        Returns actions for each agent.
        :param observations:
        :return:
        """
        pass

    def reset(self):
        pass

    def setup(self, players):
        """
        Assigns a player to each agent in the controller group
        :param players:
        :return:
        """
        roles = [player.team for player in players]
        assert np.all(np.mean(roles) == roles), "Players in a controller group should all be on the same team."
        self.players = players

    def __repr__(self):
        return self.name

class NoopController(BaseController):
    def __init__(self):
        super().__init__("noop")

    def forward(self, observations):
        return np.zeros([len(observations)], dtype=np.long)

    def setup(self, players):
        pass

class ModelController(BaseController):
    """
    Uses POAlgorithm to control agents for one role only
    """

    def __init__(self, algorithm):
        super().__init__(algorithm.name)
        self.rnn_states = None
        self.n_agents = 0
        self.algorithm = algorithm

    def forward(self, observations):
        """
        Returns actions for each agent.
        :param observations:
        :return:
        """

        assert self.players is not None, "Please call setup to assign players before forward."

        roles = torch.ones([self.n_agents], dtype=torch.long) * self.team

        model_output, new_rnn_states = self.algorithm.forward(
            obs=observations,
            rnn_states=self.rnn_states,
            roles=roles
        )
        self.rnn_states[:] = new_rnn_states

        log_policy = model_output["log_policy"].detach().cpu().numpy()
        actions = utils.sample_action_from_logp(log_policy)
        return actions

    def reset(self):
        self.rnn_states *= 0

    def setup(self, players):
        super().setup(players)
        self.n_agents = len(players)
        self.rnn_states = self.algorithm.get_initial_rnn_state(self.n_agents)
        self.team = players[0].team

class ScriptedController(BaseController):
    """
    Uses scripted response to control agents
    """

    def __init__(self, strategy, name="script"):
        super().__init__(name)
        self.strategy = strategy

    def forward(self, observations):
        """
        Returns actions for each agent. Ignores observation and uses player directly.
        :param observations:
        :return:
        """

        assert self.players is not None, "Please call setup to assign players before forward."

        # strategies ignore observations...
        actions = []
        for player in self.players:
            actions.append(self.strategy(player))
        return np.asarray(actions)

def run_evaluation(
        controllers: List[BaseController],  # controllers for each team
        scenario:str,
        log_path:str,
        trials=100,
    ):
    """
    Evalulate the performance of controllers in a given environment.

    :param controllers: list of tuples (red,green,blue)
    :param scenario: name of scenario to evaluate on
    :param log_path: path to log to 
    :param trials: number of games to run in evaluation
    :return: 
    """

    # run them all in parallel at once and make sure we get exactly 'trials' number of environments by forcing
    # them to only once (no reset)
    os.makedirs(log_path, exist_ok=True)
    vec_env = make_env(scenario, trials, name="eval", log_path=log_path)
    env_obs = vec_env.reset()

    # setup the controllers
    for team, controller in enumerate(controllers):
        controller.setup([player for player in vec_env.players if player.team == team])
        controller.reset()

    env_terminals = np.zeros([len(vec_env.players)], dtype=np.bool)
    vec_env.run_once = True

    roles = vec_env.get_roles()
    actions = np.zeros_like(roles)

    # play the game...
    results = [(0, 0, 0) for _ in range(trials)]
    while not all(env_terminals):

        actions *= 0

        # split players by team, and assign actions.
        for team, controller in enumerate(controllers):
            role_filter = (roles == team)
            actions[role_filter] = controller.forward(torch.from_numpy(env_obs)[role_filter])

        env_obs, env_rewards, env_terminals, env_infos = vec_env.step(actions)

        # look for finished games
        for i, env in enumerate(vec_env.games):
            if env.round_outcome != "":
                results[i] = env.round_team_scores

    # make sure results have be written to env log
    rescue.flush_logs()

    return results


def get_checkpoints(path):
    """
    Returns a list tuples (epoch, checkpoint_path) for given folder. Paths are full path
    :param path:
    :return:
    """
    results = []
    for f in os.listdir(path):
        if f.endswith('.pt'):
            parts = f.split('_')
            if len(parts) == 3:
                epoch = int(parts[1])
                results.append((epoch, os.path.join(path, f)))
    return results

def load_algorithm(model_path, scenario):

    # setup our algorithm
    with open(os.path.join(model_path, 'config.txt'), 'r') as f:
        kwargs = ast.literal_eval(f.read())

    kwargs["log_folder"] = '.'
    kwargs["train_scenarios"] = scenario
    kwargs["eval_scenarios"] = scenario
    kwargs["device"] = DEVICE

    config = Config()
    config.setup(kwargs)
    vec_env = make_env(scenario, parallel_envs=1) # parallel envs is for training only
    algo = make_algo(vec_env, config)
    return algo

def evaluate_vs_mixture(model_path, scenario, controller_sets: List, log_folder, title="mixture", trials=100):
    """
    Evaluate model over time vs each set of controllers in list

    :param model_path:
    :param scenario:
    :param controller_sets: A list of controllers for each team (controller_red, controller_green, controller_blue),
        any controller that is none will be replaced with the checkpoint controller
    :param trials: number of games to run per controller set per checkpoint.
    :return: List of game outcomes for each checkpoint
    """

    algo = load_algorithm(model_path, scenario)

    results = load_results(log_folder, title)
    completed_epochs = set(epoch for epoch, result, paths in results)

    for epoch, checkpoint_path in get_checkpoints(model_path):

        if epoch in completed_epochs:
            continue

        print(f"[{epoch}]: {checkpoint_path}")
        algo.load(checkpoint_path)

        # run against each set of controllers, but override red to use the checkpoint model.
        for controllers in controller_sets:
            start_time = time.time()
            # override controllers that are none
            controllers = [controller if controller is not None else ModelController(algo) for controller in controllers]
            result = run_evaluation(
                controllers=controllers,
                scenario=scenario,
                log_path=log_folder,
                trials=trials
            )
            results.append((epoch, result))
            time_taken = time.time() - start_time
            print(f" {get_mean_scores(result)} (in {time_taken:.1f}s)")

        # dump results as we go
        save_and_plot(results, log_folder, title)

    return results


def evaluate_in_parallel(
        model_paths_red: Union[str, List[str], None],
        model_paths_green: Union[str, List[str], None],
        model_paths_blue: Union[str, List[str], None],
        scenario,
        log_folder,
        title="mixture",
        trials=100,
        replace_noop_with_team = None,
        ):
    """
    Loads models for red, green, and blue from checkpoint folders and evaluates them against eachother.
    Evaluation use the cartesian product of the algorithms lists.
    If any checkpoint is missing for any model that checkpoint will be skipped for all models.

    :param model_path:
    :param scenario:
    :param controller_sets: A list of controllers for each team (controller_red, controller_green, controller_blue),
        any controller that is none will be replaced with the checkpoint controller
    :param trials: number of games to run per controller set per checkpoint.
    :return: List of game outcomes for each checkpoint
    """

    def upgrade_str_to_list(x):
        if x is None:
            return []
        return [x] if type(x) is str else x

    model_paths_red = upgrade_str_to_list(model_paths_red)
    model_paths_green = upgrade_str_to_list(model_paths_green)
    model_paths_blue = upgrade_str_to_list(model_paths_blue)

    # load algorithms
    algorithm_paths = model_paths_red + model_paths_green + model_paths_blue

    results = load_results(log_folder, title)
    completed_epochs = set(epoch for epoch, result, paths in results)

    # get checkpoints and filter down to only ones that exist in all paths
    checkpoints_lists = [get_checkpoints(algorithm_path) for algorithm_path in algorithm_paths]
    good_epochs = None
    for checkpoint_list in checkpoints_lists:
        epoch_list = set(epoch for epoch, _ in checkpoint_list)
        good_epochs = epoch_list if good_epochs is None else good_epochs.intersection(epoch_list)

    checkpoints = []
    for checkpoint_list in checkpoints_lists:
        checkpoints.append({})
        for epoch, path in checkpoint_list:
            checkpoints[-1][epoch] = path

    algorithms = []

    for epoch in sorted(good_epochs):

        if epoch in completed_epochs:
            continue

        # lazy loading of algorithms
        if len(algorithms) == 0:
            for algorithm_path in algorithm_paths:
                algo = load_algorithm(algorithm_path, scenario)
                algo.name = os.path.split(algorithm_path)[-1]
                algorithms.append(algo)

        print(f"[{epoch}]:")

        # load each algorithm at this given checkpoint
        for algo, checkpoint in zip(algorithms, checkpoints):
            algo.load(checkpoint[epoch])

        controllers = [ModelController(algo) for algo in algorithms]

        # create controllers
        red_controllers = controllers[:len(model_paths_red)]
        green_controllers = controllers[len(model_paths_red):len(model_paths_red)+len(model_paths_green)]
        blue_controllers = controllers[len(model_paths_red) + len(model_paths_green):]

        for controller in [red_controllers, green_controllers, blue_controllers]:
            if len(controller) == 0:
                controller += [NoopController()]

        # run evaluation on all combinations
        epoch_results = []
        test_sets = list(itertools.product(red_controllers, green_controllers, blue_controllers))
        for controllers in test_sets:
            controllers = list(controllers)

            # this allows two teams to use the same checkpoint (e.g. green just uses the ai from blue)
            if replace_noop_with_team is not None:
                controllers = [controller if type(controller) is not NoopController else
                               ModelController(controllers[replace_noop_with_team].algorithm)
                               for controller in controllers]

            start_time = time.time()
            # override controllers that are none
            result = run_evaluation(
                controllers=controllers,
                scenario=scenario,
                log_path=log_folder,
                trials=int(math.ceil(trials / len(test_sets)))
            )
            epoch_results += result
            time_taken = time.time() - start_time
            print(f" {get_mean_scores(result)} {controllers} (in {time_taken:.1f}s)")

        results.append((epoch, epoch_results, str(controllers)))

        # dump results as we go
        save_and_plot(results, log_folder, title)

    return results


def get_mean_scores(result_set):
    """
    Returns mean scores for each team
    :param result_set: A list of tuples (r,g,b) for each game played.
    :return: tuple (r,g,b) containing mean_scores
    """
    r = np.mean([r for r, g, b in result_set])
    g = np.mean([g for r, g, b in result_set])
    b = np.mean([b for r, g, b in result_set])
    return r,g,b

def load_results(output_folder, title):
    filename = os.path.join(output_folder, f'results_{title}.dat')
    if not os.path.exists(filename):
        return []
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def save_and_plot(results, output_folder, title):
    """
    Save a copy of the results and plot
    :param results: A list of tuples (epoch, result), where result is a list of r,g,b scores
    """

    y_r = []
    y_g = []
    y_b = []

    for epoch, result_set, paths in results:
        r,g,b = get_mean_scores(result_set)
        y_r.append(r)
        y_g.append(g)
        y_b.append(b)


    xs = [epoch for epoch, result_set, paths in results]

    plt.figure(figsize=(8, 6))

    plt.title(title)
    plt.grid(True)
    plt.plot(xs, y_r, c='red')
    plt.plot(xs, y_g, c='green')
    plt.plot(xs, y_b, c='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.savefig(os.path.join(output_folder, f'{title}.png'))
    plt.close()

    with open(os.path.join(output_folder, f'results_{title}.dat'), 'wb') as f:
        pickle.dump(results, f)


def get_all_runs(mask, exclude_mask="__"):
    all_runs = []
    for (root, dirs, files) in os.walk('./run'):
        for path in dirs:
            if mask in path and exclude_mask not in root:
                run = os.path.join(root, path)
                all_runs.append(run)
    return all_runs

def run_rescue_arena():

    control_runs = get_all_runs("413_db00")
    effect_runs = get_all_runs("413_db05")
    all_runs = control_runs + effect_runs
    print(all_runs)

    log_folder = '.\\run\\arena_rescue'

    for run in all_runs:
        run_name = os.path.split(run)[-1]
        evaluate_in_parallel(
            run, [], all_runs,
            scenario='rescue',
            log_folder=log_folder,
            title=f"red_{run_name}_vs_mixture",
            trials=TRIALS,
            replace_noop_with_team=2,
        )

    for run in all_runs:
        run_name = os.path.split(run)[-1]
        evaluate_in_parallel(
            all_runs, [], run,
            scenario='rescue',
            log_folder=log_folder,
            title=f"blue_{run_name}_vs_mixture",
            trials=TRIALS,
            replace_noop_with_team=0,
        )

    for run in all_runs:
        run_name = os.path.split(run)[-1]
        evaluate_in_parallel(
            run, [], all_runs,
            scenario='rescue_training',
            log_folder=log_folder,
            title=f"red_training_{run_name}_vs_mixture",
            trials=TRIALS,
            replace_noop_with_team=2,
        )

    for run in all_runs:
        run_name = os.path.split(run)[-1]
        evaluate_in_parallel(
            all_runs, [], run,
            scenario='rescue_training',
            log_folder=log_folder,
            title=f"blue_training_{run_name}_vs_mixture",
            trials=TRIALS,
            replace_noop_with_team=0,
        )



def main():
    """
    Check agents performance over time
    :return:
    """
    run_rescue_arena()

if __name__ == "__main__":
    main()

