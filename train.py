"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

"""
Performance notes

Model1
8-8-16 no max-pool

Model2
32-64-64 maxpool on last 2 layers

Optiplex    (CPU)  model1 = 300 FPS
My computer (CPU)  model1 = 221 FPS 
My computer (GPU)  model1 = 2100 FPS (at ~20% gpu) 
My computer (GPU)  model2 = 1300 FPS (at ~60% gpu)

"""

from rescue import RescueTheGeneralEnv
from MARL import MultiAgentVecEnv

from contextlib import nullcontext

import numpy as np
import cv2
import os
import pickle
import argparse
import time
import os

import tensorflow as tf

from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.policies import CnnLstmPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2

from new_models import single_layer_cnn

class config():
    """ Class to hold config files"""
    log_folder = str()
    device = str()


def export_video(filename, model, env):
    """
    Exports a movie with agents playing randomly.
    """

    scale = 4

    states = env.reset()
    frame = env.envs[0].render("rgb_array")

    # work out our height
    height, width, channels = frame.shape
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height), isColor=True)

    dones = [False] * len(states)

    team_scores = np.zeros([3], dtype=np.int)

    # don't like it that this is hard coded... not sure how to init the states?
    agent_states = np.zeros((len(states), 512))

    n_players = env.envs[0].n_players

    # play the game...
    while not np.all(dones[:n_players]):

        stacked_states = np.asarray(states)
        actions, _, agent_states, _ = model.step(stacked_states, agent_states, dones)

        states, rewards, dones, infos = env.step(actions)

        for i in range(n_players):
            team_scores[env.envs[0].player_team[i]] += rewards[i]

        # generate frames from global perspective
        frame = env.envs[0].render("rgb_array")

        # for some reason cv2 wants BGR instead of RGB
        frame[:, :, :] = frame[:, :, ::-1]

        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format((width, height, 3), frame.shape)
        video_out.write(frame)

    for _ in range(15):
        # this just makes it easier to see the last frame on some players
        video_out.write(frame*0)

    video_out.release()

    return team_scores

def train_model():
    """
    Train PPO on the environment using the "other agents are environment" method.
    :return:
    """

    print("Starting environment")

    # copy source files for later
    from shutil import copyfile
    for filename in ["train.py", "rescue.py"]:
        copyfile(filename, f"{config.log_folder}/{filename}")

    # our MARL environments are handled like vectorized environments
    vec_env = MultiAgentVecEnv([RescueTheGeneralEnv for _ in range(16)])

    model = PPO2(CnnLstmPolicy, vec_env, verbose=1, learning_rate=2.5e-4, ent_coef=0.001,
         policy_kwargs={"cnn_extractor": single_layer_cnn})

    for sub_env in vec_env.envs:
        sub_env.log_folder = config.log_folder

    # special case for first epoch, we want some early results
    for mini_epoch in range(0, 10):
        print("Generating video...")
        export_video(f"{config.log_folder}/ppo_run_000_{mini_epoch}_M.mp4", model, vec_env)
        if mini_epoch == 0:
            model.save(f"{config.log_folder}/model_000_M.p")
        print("Training...")
        model.learn(100000, reset_num_timesteps=False, log_interval=10)

    for epoch in range(1, 100):
        print("Generating video...")
        export_video(f"{config.log_folder}/ppo_run_{epoch:03}M.mp4", model, vec_env)
        model.save(f"{config.log_folder}/model_{epoch:03}M.p")
        print("Training...")
        model.learn(1000000, reset_num_timesteps=False, log_interval=10)

    model.save(f"{config.log_folder}/model_final.p")

    print("Finished training.")

def generate_video():
    """
    Generate a video of random play then exit.
    """

    print("Generating video.")
    env = MultiAgentVecEnv([RescueTheGeneralEnv])
    model = PPO2(CnnPolicy, env, verbose=1, learning_rate=2.5e-4, ent_coef=0.001, policy_kwargs={"cnn_extractor":single_layer_cnn})
    export_video(f"{config.log_folder}/ppo_run-0.mp4", model, env)

def run_benchmark():

    print("Benchmarking environment")

    env = RescueTheGeneralEnv()

    start_time = time.time()

    env.reset()
    for _ in range(10000):
        random_actions = np.random.randint(0,10, size=(env.n_players,))
        env.step(random_actions)

    time_taken_ms = (time.time() - start_time)

    print(f"Native environment runs at {10000/time_taken_ms:.0f} FPS.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="[bench|video|train]")
    parser.add_argument('--run', type=str, help="run folder", default="test")
    parser.add_argument('--device', type=str, help="[0|1|2|3|AUTO]", default="auto")

    args = parser.parse_args()

    # setup config
    config.log_folder = f"run/{args.run}"
    config.device = args.device

    if config.device.lower() != "auto":
        print(f"Using GPU {config.device}")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    os.makedirs(config.log_folder, exist_ok=True)

    if args.mode == "bench":
        run_benchmark()
    elif args.mode == "video":
        generate_video()
    elif args.mode == "train":
        train_model()
    else:
        raise Exception(f"Invalid mode {args.mode}")

if __name__ == "__main__":
    main()