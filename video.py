# Video exporter for use with rescue game

import os
import cv2
import utils
import torch
import numpy as np
import shutil

from typing import Union

from rescue import RescueTheGeneralEnv
from train import make_env
from algorithms import PMAlgorithm
from utils import draw_pixel

def display_role_prediction(frame: np.ndarray, dx: int, dy: int, raw_predictions, env:RescueTheGeneralEnv):
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

def display_policy(frame: np.ndarray, dx:int, dy:int, policy: np.ndarray,
                   action: Union[int, None] = None,
                   prev_action: Union[int, None] = None,

                   ):
    """
    :param frame:
    :param dx:
    :param dy:
    :param policy: nd array of dims [n_roles, n_actions] as probability distribution (0..1)
    :param action: (optional) the action the player took
    :return:
    """
    n_roles, n_actions = policy.shape

    base_colors = np.asarray([
        (0.3, 0.3, 0.3),  # no-op
        (0.1, 0.1, 0.1),  # move
        (0.1, 0.1, 0.1),  # move
        (0.1, 0.1, 0.1),  # move
        (0.1, 0.1, 0.1),  # move
        (0.3, 0.3, 0.3),  # act
        (0.1, 0.1, 0.1),  # shoot
        (0.1, 0.1, 0.1),  # shoot
        (0.1, 0.1, 0.1),  # shoot
        (0.1, 0.1, 0.1),  # shoot

    ])

    for r in range(n_roles):
        on_color = np.asarray((0.0, 0.0, 0.0))
        on_color[r] = 1.0
        for a in range(n_actions):
            weight = policy[r, a]
            c = weight * on_color
            frame[dy + r, dx + a] = (c*255).astype(np.uint8)

    for a in range(n_actions):
        # show markers at bottom indicating what is what
        if a < len(base_colors):
            c = base_colors[a]
        else:
            c = 0.0
        if action is not None and a == action:
            c = 0.8 # indicate which action the player took.
        elif prev_action is not None and a == prev_action:
            c = 0.5 # show the previous action too, if required (useful for deception bonus)

        c = np.asarray(c)

        frame[dy + n_roles, dx + a] = (c * 255).astype(np.uint8)

def export_video(filename, algorithm: PMAlgorithm, scenario):
    """
    Exports a movie of model playing in given scenario
    """

    scale = 8
    n_roles = 3

    # make a new environment so we don't mess the settings up on the one used for training.
    # it also makes sure that results from the video are not included in the log
    vec_env = make_env(scenario, parallel_envs=1, name="video")

    env_obs = vec_env.reset()
    env = vec_env.games[0]
    frame = env.render("rgb_array")

    obs_size = env.observation_space.shape[1]  # stub, make this a shape and get from env
    n_players = len(env.players)
    n_actions = vec_env.action_space.n

    # work out our height
    height, width, channels = frame.shape
    orig_height, orig_width = height, width

    if algorithm.uses_deception_model:

        if algorithm.predicts_observations:
            prediction_display_width = obs_size
            prediction_display_height = obs_size
        else:
            prediction_display_width = (n_actions + 1) + 8
            prediction_display_height = (n_roles + 1) + 8

        # make room for predictions
        role_prediction_space = (n_players + 2) * 8 * 2
        width = max(width + role_prediction_space , (n_players * 2 + 1) * prediction_display_width)

        # make room for other predictions
        if algorithm.predicts_observations:
            height = height + n_players * prediction_display_height
        if algorithm.predicts_actions:
            height = height + n_players * prediction_display_height

    scaled_width = (width * scale) // 4 * 4  # make sure these are multiples of 4
    scaled_height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 8, (scaled_width, scaled_height),
                                isColor=True)

    # this is required to make sure the last frame is visible
    vec_env.auto_reset = False

    rnn_state = algorithm.get_initial_rnn_state(len(env.players))

    last_outcome = ""
    bonus = None

    def channels_first_to_last(x):
        """
        Swap from chw to whc
        :param x:
        :return:
        """
        return x.swapaxes(0, 2)

    actions = np.zeros([len(vec_env.players)], dtype=np.long)

    # play the game...
    while last_outcome == "":

        last_outcome = env.round_outcome
        prev_actions = actions.copy()

        with torch.no_grad():
            roles = vec_env.get_roles()
            obs_truth = env_obs.copy()
            model_output, new_rnn_state = algorithm.forward(
                torch.from_numpy(env_obs),
                rnn_state,
                torch.from_numpy(roles)
            )

            rnn_state[:] = new_rnn_state

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
            display_role_prediction(frame, orig_width, 10, role_predictions, env)

        if "policy_role_prediction" in model_output:
            role_predictions = model_output["policy_role_prediction"].detach().cpu().numpy()
            display_role_prediction(frame, orig_width, 60, role_predictions, env)

        if "role_backwards_prediction" in model_output:
            backwards_role_predictions = model_output["role_backwards_prediction"].detach().cpu().numpy()
            backwards_role_predictions_transposed = backwards_role_predictions.swapaxes(0, 1)
            display_role_prediction(frame, orig_width + (n_players + 2) * 8, 10, backwards_role_predictions_transposed, env)

        if "obs_prediction" in model_output:
            # ground truth
            for i in range(n_players):
                dx = 0
                dy = orig_height + i * obs_size
                frame[dy:dy + obs_size, dx:dx + obs_size] = channels_first_to_last(obs_truth[i])

            # predictions
            # observation frames are [n_players, n_predictions, c, h, w]
            obs_predictions = model_output["obs_prediction"].detach().cpu().numpy()
            n_players, n_predictions, c, h, w = obs_predictions.shape
            for i in range(n_players):
                for j in range(n_predictions):
                    dx = j * obs_size + obs_size
                    dy = orig_height + i * obs_size
                    predictions_transposed = channels_first_to_last(np.asarray(obs_predictions[i, j] * 255, dtype=np.uint8))
                    frame[dy:dy + obs_size, dx:dx + obs_size] = predictions_transposed

        if "obs_backwards_prediction" in model_output:
            obs_pp = model_output["obs_backwards_prediction"].detach().cpu().numpy()
            for i in range(n_players):
                for j in range(n_players):
                    dx = j * obs_size + (obs_size * (n_players + 1))
                    dy = orig_height + i * obs_size
                    # we transpose as rescue is x,y instead of y,x
                    predictions_transposed = channels_first_to_last(np.asarray(obs_pp[j, i] * 255, dtype=np.uint8))
                    frame[dy:dy + obs_size, dx:dx + obs_size] = predictions_transposed

        if "action_prediction" in model_output:
            action_predictions = np.exp(model_output["action_prediction"].detach().cpu().numpy())
            action_prediction_predictions = np.exp(model_output["action_backwards_prediction"].detach().cpu().numpy())
            true_policy = np.exp(model_output["role_log_policy"].detach().cpu().numpy())
            _, _, _, n_actions = action_predictions.shape

            # these come out as n_players, n_players, n_roles, n_actions ?
            policy_spacing_y = (n_roles+2)

            # true policy
            for i in range(n_players):
                dx = 0 * (n_actions+1) + 4
                dy = orig_height + i * policy_spacing_y
                display_policy(frame, dx, dy, true_policy[i], actions[i], prev_actions[i])

            # predicted policy
            for i in range(n_players):
                for j in range(n_players):
                    dx = (j+1) * (n_actions+1) + 8
                    dy = orig_height + i * policy_spacing_y
                    display_policy(frame, dx, dy, action_predictions[i, j])

            # predictions of other players predictions of our own policy
            for i in range(n_players):
                for j in range(n_players):
                    dx = (n_players+(j+1)) * (n_actions+1) + 12
                    dy = orig_height + i * policy_spacing_y
                    display_policy(frame, dx, dy, action_prediction_predictions[i, j])

        # add deception bonus indicators (on top of role prediction)
        if bonus is not None:

            for i in range(n_players):
                dx = orig_width
                dy = 10 + (i + 1) * 8
                for scale in [0, 1, 2]:

                    factor = bonus[i] / (10 ** (2 - scale))

                    if factor > 0:
                        c = np.asarray((factor, 0.0, 0.0), np.float)
                    else:
                        c = np.asarray((0.0, 0.0, -factor), np.float)

                    c = np.clip(c * 255, 0, 255).astype(np.uint8)

                    frame[dy+scale+1:dy+8-scale-1, dx+scale+1:dx+8-scale-1] = c

        # write time stamp
        utils.draw_numbers(frame, width-4*5, height-6, str(env.round_timer), [255, 255, 255], zero_pad=4)

        # for some reason cv2 wants BGR instead of RGB
        frame[:, :, :] = frame[:, :, ::-1]

        if frame.shape[0] != scaled_width or frame.shape[1] != scaled_height:
            frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

        assert \
            frame.shape[1] == scaled_width and frame.shape[0] == scaled_height, \
            "Frame should be {} but is {}".format((scaled_width, scaled_height, 3), frame.shape)

        video_out.write(frame)

        # step environment
        if last_outcome == "":
            env_obs, env_rewards, env_dones, env_infos = vec_env.step(actions)

        # calculate deception bonus
        if algorithm.uses_deception_model and not algorithm.predicts_observations:
            # show bonus, only works on actions at the moment
            # this bonus is for the action we *will* take on the next frame.
            players_visible = []
            for player in env.players:
                vision = [player.in_vision(other_player.x, other_player.y) for other_player in env.players]
                players_visible.append(vision)
            players_visible = np.asarray(players_visible)
            bonus = algorithm.calculate_deception_bonus(model_output, actions, vec_env, roles, players_visible)
        else:
            bonus = None

    # write last frame out 10 times
    # this is because some video players terminate at the end of the video making it *very* hard to see the last frame
    # (which is often very important)
    for _ in range(10):
        video_out.write(frame)

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



