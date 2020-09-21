"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

LOG_FOLDER = "Experiment_6"

from rescue import RescueTheGeneralEnv, MultiAgentEnvAdapter

import numpy as np
import cv2
import os
import pickle

import gym

from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.policies import CnnLstmPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2

from new_models import single_layer_cnn

def export_movie(filename, model):

    """ Exports a movie with agents playing randomly.
        which_frames: model, real, or both
    """

    scale = 4

    env = RescueTheGeneralEnv()
    states = env.reset()
    frame = env.render("rgb_array")

    # work out our height
    height, width, channels = frame.shape
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height), isColor=True)

    dones = [False] * env.n_players

    team_scores = np.zeros([3], dtype=np.int)

    # play the game...
    while not np.all(dones):

        # todo: implement LSTM
        actions, _, _, _ = model.step(np.asarray(states), None, None)

        states, rewards, dones, infos = env.step(actions)

        for i in range(env.n_players):
            team_scores[env.player_team[i]] += rewards[i]

        frame = env.render("rgb_array", player_id=0) # stub, view from player 0's perspective.

        # for some reason cv2 wants BGR instead of RGB
        frame[:, :, :] = frame[:, :, ::-1]

        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format((width, height, 3), frame.shape)

    for _ in range(4):
        # this just makes it easier to see the last frame
        video_out.write(frame)

    video_out.release()

    return team_scores

def make_env():
    env = RescueTheGeneralEnv()
    env = MultiAgentEnvAdapter(env)
    return env

def play_simple_game():
    """
    Train PPO on the environment using the "other agents are environment" method.
    :return:
    """

    print("Starting environment")

    # mutli-processor not supported yet. Would require sending the model to each process, and I don't know if
    # tensorflow allows instances running across processes like that.
    vec_env = DummyVecEnv([make_env for _ in range(16)])
    #vec_env = VecNormalize(vec_env, norm_obs=False, clip_obs=False, norm_reward=True, clip_reward=False)

    # create model
    model = PPO2(CnnPolicy, vec_env, verbose=1, ent_coef=0.001, policy_kwargs={"cnn_extractor":single_layer_cnn})

    for sub_env in vec_env.envs:
        sub_env.model = model
        sub_env.env.log_folder = LOG_FOLDER

    scores = []

    scores.append(export_movie(f"{LOG_FOLDER}/ppo_run-0.mp4", model))

    # this is mostly just to see how big the model is..
    model.save(f"{LOG_FOLDER}/model_initial.p")

    for epoch in range(1000):

        model.learn(100000, reset_num_timesteps=epoch==0, log_interval=10)

        print("Generating movie...")
        scores.append(export_movie(f"{LOG_FOLDER}/ppo_run-{epoch+1}.mp4", model))

        pickle.dump(scores, open(f"{LOG_FOLDER}/results.dat", "wb"))

    print("Finished training.")

def main():

    os.makedirs(LOG_FOLDER, exist_ok=True)

    print("Rendering movie...")
    play_simple_game()
    print("Done.")

if __name__ == "__main__":
    main()