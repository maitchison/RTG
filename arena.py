"""
Fight agents against eachother, and against scripted responses
"""

from typing import List, Union
from algorithms import PMAlgorithm
from train import make_env
from marl_env import MultiAgentVecEnv

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

def load_checkpoint(path, epoch:Union[int, None]=None):
    """
    Loads a model from a checkpoint file (.pt) and returns it
    :param path:
    :return:
    """

    with open(os.path.join(path, 'config.txt'), 'rb') as f:
        kwargs = ast.literal_eval(f.readlines())
    algo = PMAlgorithm(**kwargs)

    if epoch is not None:
        epoch_str = f'_{epoch}_M'
    else:
        epoch_str = ''
    algo.load(os.path.join(path, f'model{epoch_str}.pt'))


def play_game(env: MultiAgentVecEnv, red_controller, green_controller, blue_controller):
    """
    Plays a game with the given controllers. The environment is vectorized so multiple games can be played in parallel.
    Returns the outcome of the game. Optionally a video can be recorded.
    :param env:
    :param red_controller:
    :param green_controller:
    :param blue_controller:
    :return:
    """
    pass

def run_arena(controllers: List[Controller]):
    """
    Run an arena with the given controllers.

    Each controller is run against an equal mixture of the other controllers,
    For red and blue their pairwise average scores are recorded, as well as their overall average score.

    :param controlers:
    :return: average score for each controler, and pairwise scores for each red/blue combination
    """

    red_controllers = []
    green_controllers = []
    blue_controllers = []

    env = ...

    for red in red_controllers:
        for blue in blue_controllers:
            for green in green_controllers:
                result[(red, blue)] += play_game(env, red, green, blue)


def run_evaluation(output_path, algorithm:PMAlgorithm, scenario:str, trials=100,):
    """
    :param model_path: path to model
    :param scenario: name of scenario to test on.
    :return: a list of tuples containing trials game scores.
    """

    # run them all in parallel at once and make sure we get exactly 'trials' number of environments by forcing
    # them to only once (no reset)
    os.makedirs(output_path, exist_ok=True)
    vec_env = make_env(scenario, name="eval", log_path=output_path, parallel_envs=trials)
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

    # collate results
    red_score = np.mean([r for r, g, b in results])
    green_score = np.mean([g for r, g, b in results])
    blue_score = np.mean([b for r, g, b in results])

    # make sure results have be written to env log
    rescue.flush_logs()


def get_checkpoints(path):
    """
    Returns a list tuples (epoch, checkpoint_path) for given folder. Paths are full path
    :param path:
    :return:
    """
    pass

def evaluate_vs_self(model_path, scenario):
    """
    Evaluate model against itself over time in given environment

    :param model_path:
    :param scenario:
    :return:
    """

    sub_folder = os.path.join(model_path, 'arena')

    results = []

    for epoch, checkpoint_path in get_checkpoints(model_path):
        algo = load_checkpoint(checkpoint_path)
        result = run_evaluation(sub_folder, algo, scenario)
        results.append((epoch, result))

    return results

def save_and_plot(results, output_folder, title):
    """
    Save a copy of the results and plot
    :param results:
    """
    xs = [epoch for epoch, (r,g,b) in results]
    y_r = [r for epoch, (r,g,b) in results]
    y_g = [g for epoch, (r, g, b) in results]
    y_b = [b for epoch, (r, g, b) in result]

    plt.figure(figsize=(8, 6))

    plt.title(title)
    plt.plot(xs, y_r)
    plt.plot(xs, y_g)
    plt.plot(xs, y_b)

    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.savefig(os.path.join(output_folder, f'{title}.png'))


def main():
    """
    Check agents performance over time
    :return:
    """

    run_path = './run/exp49a_alt_off'

    # create an environment with 100 games for testing
    vec_env = make_env('alta', 100)

    # evaluate against ourself over time
    save_and_plot(evaluate_vs_self(run_path, 'alta'), run_path, 'Vs Self')


    # evaluate against each other over time (with green being 50/50)

    # evaluate against scripted responses over time

    # run arena at various points in time


