"""
Fight agents against eachother, and against scripted responses
"""

from typing import List, Union
from algorithms import PMAlgorithm
from support import make_env, make_algo, Config
from marl_env import MultiAgentVecEnv
import rescue
import pickle

import ast
import os
import torch
import utils
import matplotlib.pyplot as plt

import numpy as np

class Controller():
    """
    Controller for game, plays one side only
    """
    pass

def load_checkpoint(path) -> PMAlgorithm:
    """
    Loads a model from a checkpoint file (.pt) and returns it
    :param path:
    :return:
    """
    with open(os.path.join(os.path.split(path)[0], 'config.txt'), 'r') as f:
        kwargs = ast.literal_eval(f.read())

    scenario = kwargs["eval_scenarios"][0][0]

    kwargs["log_folder"] = '.'
    kwargs["train_scenarios"] = scenario
    kwargs["eval_scenarios"] = scenario

    config = Config()
    config.setup(kwargs)

    vec_env = make_env(scenario, config.parallel_envs)
    algo = make_algo(vec_env, config)

    algo.load(path)
    return algo

# def play_game(env: MultiAgentVecEnv, red_controller, green_controller, blue_controller):
#     """
#     Plays a game with the given controllers. The environment is vectorized so multiple games can be played in parallel.
#     Returns the outcome of the game. Optionally a video can be recorded.
#     :param env:
#     :param red_controller:
#     :param green_controller:
#     :param blue_controller:
#     :return:
#     """
#     pass

# def run_arena(controllers: List[Controller]):
#     """
#     Run an arena with the given controllers.
#
#     Each controller is run against an equal mixture of the other controllers,
#     For red and blue their pairwise average scores are recorded, as well as their overall average score.
#
#     :param controlers:
#     :return: average score for each controler, and pairwise scores for each red/blue combination
#     """
#
#     red_controllers = []
#     green_controllers = []
#     blue_controllers = []
#
#     env = ...
#
#     for red in red_controllers:
#         for blue in blue_controllers:
#             for green in green_controllers:
#                 result[(red, blue)] += play_game(env, red, green, blue)

def run_evaluation(algorithm:PMAlgorithm, scenario:str, log_path:str, trials=100):
    """
    :param model_path: path to model
    :param scenario: name of scenario to test on.
    :return: a list of tuples containing trials game scores.
    """

    # run them all in parallel at once and make sure we get exactly 'trials' number of environments by forcing
    # them to only once (no reset)
    os.makedirs(log_path, exist_ok=True)
    vec_env = make_env(scenario, trials, name="eval", log_path=log_path)
    env_obs = vec_env.reset()

    rnn_states = algorithm.get_initial_rnn_state(vec_env.num_envs)
    env_terminals = np.zeros([len(rnn_states)], dtype=np.bool)
    vec_env.run_once = True

    # play the game...
    results = [(0, 0, 0) for _ in range(trials)]
    while not all(env_terminals):

        with torch.no_grad():
            roles = vec_env.get_roles()
            model_output, new_rnn_states = algorithm.forward(
                obs=torch.from_numpy(env_obs),
                rnn_states=rnn_states,
                roles=torch.from_numpy(roles)
            )
            rnn_states[:] = new_rnn_states

            log_policy = model_output["log_policy"].detach().cpu().numpy()
            actions = utils.sample_action_from_logp(log_policy)

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

def evaluate_vs_self(model_path, scenario):
    """
    Evaluate model against itself over time in given environment

    :param model_path:
    :param scenario:
    :return:
    """

    log_folder = os.path.join(model_path, 'arena')

    results = []

    for epoch, checkpoint_path in get_checkpoints(model_path):
        print(f"[{epoch}]: {checkpoint_path}")
        algo = load_checkpoint(checkpoint_path)
        result = run_evaluation(algo, scenario, log_folder)
        results.append((epoch, result))
        print(f" -{get_mean_scores(result)}")

        # dump results as we go
        save_and_plot(results, log_folder, "Vs Self")

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


def save_and_plot(results, output_folder, title):
    """
    Save a copy of the results and plot
    :param results: A list of tuples (epoch, result), where result is a list of r,g,b scores
    """

    y_r = []
    y_g = []
    y_b = []

    for epoch, result_set in results:
        r,g,b = get_mean_scores(result_set)
        y_r.append(r)
        y_g.append(g)
        y_b.append(b)


    xs = [epoch for epoch, result_set in results]

    plt.figure(figsize=(8, 6))

    plt.title(title)
    plt.grid(True)
    plt.plot(xs, y_r, c='red')
    plt.plot(xs, y_g, c='green')
    plt.plot(xs, y_b, c='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.savefig(os.path.join(output_folder, f'{title}.png'))

    with open(os.path.join(output_folder, 'results.dat'), 'wb') as f:
        pickle.dump(results, f)



def main():
    """
    Check agents performance over time
    :return:
    """

    run_path = './run/rescue410a_off [10ddd00d]'

    # evaluate against ourself over time
    evaluate_vs_self(run_path, 'rescue_a')

    # evaluate against each other over time (with green being 50/50)

    # evaluate against scripted responses over time

    # run arena at various points in time

if __name__ == "__main__":
    main()

