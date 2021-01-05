# Video exporter for use with rescue game

import os
import cv2
import utils
import torch
import numpy as np
import shutil

from rescue import RescueTheGeneralEnv
from train import make_env
from algorithms import PMAlgorithm
from utils import draw_pixel

def _display_role_prediction(frame: np.ndarray, dx: int, dy: int, raw_predictions, env:RescueTheGeneralEnv):
    """
    :param dx: draw location
    :param dy: draw location
    :param raw_predictions: nd array of dims [n_players, n_players, n_roles]
    :return:
    """

    n_players = len(raw_predictions)

    # format the role predictions
    role_predictions = np.exp(raw_predictions)

    block_size = 8
    for i in range(n_players):
        draw_pixel(frame, dy, dx + (i + 1) * block_size, c=env.players[i].team_color,
                   size=block_size)  # indicate roles
        draw_pixel(frame, dy + (i + 1) * block_size, dx, c=env.players[i].id_color,
                   size=block_size)  # indicate roles

    for i in range(n_players):
        for j in range(n_players):
            c = [int(x * 255) for x in role_predictions[i, j]]
            draw_pixel(frame, dy + (i + 1) * block_size, dx + (j + 1) * block_size, c=c, size=block_size)

def export_video(filename, algorithm: PMAlgorithm, scenario):
    """
    Exports a movie of model playing in given scenario
    """

    scale = 8

    # make a new environment so we don't mess the settings up on the one used for training.
    # it also makes sure that results from the video are not included in the log
    vec_env = make_env(scenario, parallel_envs=1, name="video")

    env_obs = vec_env.reset()
    env = vec_env.games[0]
    frame = env.render("rgb_array")

    obs_size = env.observation_space.shape[1]  # stub, make this a shape and get from env
    n_players = len(env.players)

    # work out our height
    height, width, channels = frame.shape
    orig_height, orig_width = height, width
    if algorithm.uses_deception_model:
        height = height + n_players * obs_size
        width = max(width, (n_players * 2 + 1) * obs_size)
    scaled_width = (width * scale) // 4 * 4  # make sure these are multiples of 4
    scaled_height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 8, (scaled_width, scaled_height),
                                isColor=True)

    # this is required to make sure the last frame is visible
    vec_env.auto_reset = False

    rnn_state = algorithm.get_initial_rnn_state(len(env.players))

    # play the game...
    while env.round_outcome == "":

        with torch.no_grad():
            roles = vec_env.get_roles()
            obs_truth = env_obs.copy()
            model_output, new_rnn_state = algorithm.forward(env_obs, rnn_state, roles)

            rnn_state[:] = new_rnn_state[:]

            log_policy = model_output["log_policy"].detach().cpu().numpy()
            actions = utils.sample_action_from_logp(log_policy)

        # generate frames from global perspective
        frame = env.render("rgb_array")

        blank_frame = np.zeros([height, width, 3], dtype=np.uint8)
        blank_frame[:frame.shape[0], :frame.shape[1], :] = frame  # copy into potentially larger frame
        frame = blank_frame

        # add additional parts based on the output of the model
        if "role_prediction" in model_output:
            role_predictions = model_output["role_prediction"].detach().cpu().numpy()
            _display_role_prediction(frame, orig_width, 10, role_predictions, env)

        if "role_backwards_prediction" in model_output:
            backwards_role_predictions = model_output["role_backwards_prediction"].detach().cpu().numpy()
            _display_role_prediction(frame, orig_width + (n_players+2) * 8, 10, backwards_role_predictions, env)

        if "obs_prediction" in model_output:
            # ground truth
            for i in range(n_players):
                dx = 0
                dy = orig_height + i * obs_size
                frame[dy:dy + obs_size, dx:dx + obs_size] = obs_truth[i].swapaxes(0, 1)

            # predictions
            # observation frames are [n_players, n_predictions, h, w, c]
            obs_predictions = model_output["obs_prediction"].detach().cpu().numpy()
            n_players, n_predictions, h, w, c = obs_predictions.shape
            for i in range(n_players):
                for j in range(n_predictions):
                    dx = j * obs_size + obs_size
                    dy = orig_height + i * obs_size
                    # we transpose as rescue is x,y instead of y,x
                    frame[dy:dy + obs_size, dx:dx + obs_size] = \
                        np.asarray(obs_predictions[i, j] * 255, dtype=np.uint8).swapaxes(0, 1)

        if "obs_backwards_prediction" in model_output:
            obs_pp = model_output["obs_backwards_prediction"].detach().cpu().numpy()
            for i in range(n_players):
                for j in range(n_players):
                    dx = j * obs_size + (obs_size * (n_players + 1))
                    dy = orig_height + i * obs_size
                    # we transpose as rescue is x,y instead of y,x
                    frame[dy:dy + obs_size, dx:dx + obs_size] = \
                        np.asarray(obs_pp[i, j] * 255, dtype=np.uint8).swapaxes(0, 1)

        if "action_prediction" in model_output:
            action_predictions = np.exp(model_output["action_prediction"].detach().cpu().numpy())
            action_prediction_predictions = np.exp(model_output["action_backwards_prediction"].detach().cpu().numpy())
            true_policy = np.exp(model_output["role_log_policy"].detach().cpu().numpy())

            _, _, n_roles, n_actions = action_predictions.shape

            # these come out as n_players, n_players, n_roles, n_actions ?

            # true policy
            for i in range(n_players):
                dx = 0 * (n_actions+1)
                dy = orig_height + i * (n_roles+1)
                for r in range(n_roles):
                    # i's belief about j's actions if they were role r
                    for a in range(n_actions):
                        frame[dy + r, dx + a, :] = 64
                        frame[dy + r, dx + a, r] = int(255*true_policy[i, r, a])

            # predicted policy
            for i in range(n_players):
                for j in range(n_players):
                    dx = (j+1) * (n_actions+1)
                    dy = orig_height + i * (n_roles+1)
                    for r in range(n_roles):
                        # i's belief about j's actions if they were role r
                        for a in range(n_actions):
                            frame[dy + r, dx + a, :] = 64
                            frame[dy + r, dx + a, r] = int(255*action_predictions[i, j, r, a])

            # predictions of other players predictions of our own policy
            for i in range(n_players):
                for j in range(n_players):
                    dx = (n_players+j+2) * (n_actions+1)
                    dy = orig_height + i * (n_roles+1)
                    for r in range(n_roles):
                        # i's belief about j's actions if they were role r
                        for a in range(n_actions):
                            frame[dy + r, dx + a, :] = 64
                            frame[dy + r, dx + a, r] = int(255*action_prediction_predictions[i, j, r, a])


        # for some reason cv2 wants BGR instead of RGB
        frame[:, :, :] = frame[:, :, ::-1]

        if frame.shape[0] != scaled_width or frame.shape[1] != scaled_height:
            frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

        assert \
            frame.shape[1] == scaled_width and frame.shape[0] == scaled_height, \
            "Frame should be {} but is {}".format((scaled_width, scaled_height, 3), frame.shape)

        video_out.write(frame)

        # step environment
        env_obs, env_rewards, env_dones, env_infos = vec_env.step(actions)

    video_out.release()

    # rename video to include outcome
    try:
        outcome = env.game_outcomes
        # if we only ran one round then remove the unnecessary array brackets
        if type(outcome) == list and len(outcome) == 1:
            outcome = outcome[0]
        modified_filename = f"{os.path.splitext(filename)[0]} [{outcome}]{os.path.splitext(filename)[1]}"
        shutil.move(filename, modified_filename)
    except:
        print("Warning: could not rename video file.")



