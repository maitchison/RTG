"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

# disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # errors only
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from rescue import RescueTheGeneralEnv
from MARL import MultiAgentVecEnv

from contextlib import nullcontext

import numpy as np
import cv2
import os
import argparse
import time

from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.policies import CnnLstmPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2

from new_models import cnn_default, cnn_fast

class config():
    """ Class to hold config files"""
    log_folder = str()
    device = str()
    scenario = str()
    epochs = int()
    model_name = str()

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

    # don't like it that this is hard coded... not sure how to init the states?
    agent_states = model.initial_state

    n_players = env.envs[0].n_players

    # play the game...
    while not np.all(dones[:n_players]):

        stacked_states = np.asarray(states)
        actions, _, agent_states, _ = model.step(stacked_states, agent_states, dones)

        states, rewards, dones, infos = env.step(actions)

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

def train_model():
    """
    Train PPO on the environment using the "other agents are environment" method.
    :return:
    """

    print("=============================================================")

    print("Starting environment")

    # copy source files for later
    from shutil import copyfile
    for filename in ["train.py", "rescue.py"]:
        copyfile(filename, f"{config.log_folder}/{filename}")

    # our MARL environments are handled like vectorized environments
    vec_env = MultiAgentVecEnv([lambda: RescueTheGeneralEnv(scenario=config.scenario) for _ in range(16)])
    print("Scenario parameters:")
    print(vec_env.envs[0].scenario)

    model = make_model(vec_env)

    for sub_env in vec_env.envs:
        sub_env.log_folder = config.log_folder

    print("=============================================================")

    # special case for first epoch, we want some early results
    for mini_epoch in range(0, 10):
        print("Generating video...")
        export_video(f"{config.log_folder}/ppo_run_000_{mini_epoch}_M.mp4", model, vec_env)
        if mini_epoch == 0:
            model.save(f"{config.log_folder}/model_000_M.p")
        print("Training...")
        model.learn(100000, reset_num_timesteps=False, log_interval=10)

    for epoch in range(1, config.epochs):
        print("Generating video...")
        export_video(f"{config.log_folder}/ppo_run_{epoch:03}_M.mp4", model, vec_env)
        model.save(f"{config.log_folder}/model_{epoch:03}_M.p")
        print(f"Training epoch {epoch} on experiment {config.log_folder}")
        model.learn(1000000, reset_num_timesteps=False, log_interval=10)

    model.save(f"{config.log_folder}/model_final.p")

    # flush the log buffer
    for env in vec_env.envs:
        env.write_log_buffer()

    print("Finished training.")

def make_model(env, model_name = None):

    model_name = model_name or config.model_name

    if model_name == "cnn_lstm_default":
        return PPO2(CnnLstmPolicy, env, verbose=1, learning_rate=2.5e-4, ent_coef=0.001, n_steps=128,
                    n_cpu_tf_sess=1,    # limiting cpu count really helps performance a lot when using GPU
                    policy_kwargs={
                        "cnn_extractor": cnn_default,
                        "n_lstm": 256
                    })
    elif model_name == "cnn_lstm_fast":
        return PPO2(CnnLstmPolicy, env, verbose=1, learning_rate=2.5e-4, ent_coef=0.001, n_steps=128,
                    n_cpu_tf_sess=1,
                    policy_kwargs={
                        "cnn_extractor": cnn_fast,
                        "n_lstm": 128
                    })
    else:
        raise ValueError(f"Invalid model name {model_name}")



def run_benchmark():

    print("Benchmarking environment...")
    vec_env = MultiAgentVecEnv([RescueTheGeneralEnv for _ in range(16)])

    states = vec_env.reset()
    steps = 0

    start_time = time.time()

    while time.time() - start_time < 10:
        random_actions = np.random.randint(0, 10, size=[vec_env.num_envs])
        states, _, _, _ = vec_env.step(random_actions)
        steps += 16

    time_taken = (time.time() - start_time)

    print(f" - environment runs at {steps/time_taken:.0f} FPS.")

    print("Benchmarking models...")

    def bench_model(model, model_name):

        model_states = model.initial_state
        model_masks = np.zeros((vec_env.num_envs,), dtype=np.uint8)
        steps = 0

        start_time = time.time()

        while time.time() - start_time < 10:

            actions, _, model_states, _ = model.step(np.asarray(states), model_states, model_masks)

            steps += 16

        time_taken = (time.time() - start_time)

        print(f" - model {model_name} runs at {steps / time_taken:.0f} FPS.")

    for model_name in ["cnn_lstm_default", "cnn_lstm_fast"]:
        bench_model(make_model(vec_env, model_name), model_name)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="[bench|train]")
    parser.add_argument('--run', type=str, help="run folder", default="test")
    parser.add_argument('--device', type=str, help="[0|1|2|3|AUTO]", default="auto")
    parser.add_argument('--scenario', type=str, help="[default|red2]", default="default")
    parser.add_argument('--epochs', type=int, help="number of epochs to train for (each 1M agent steps)", default=100)
    parser.add_argument('--model', type=str, help="model to use [cnn_lstm_default|cnn_lstm_fast]", default="cnn_lstm_default")

    args = parser.parse_args()

    # setup config
    config.log_folder = f"run/{args.run}"
    config.device = args.device
    config.scenario = args.scenario
    config.epochs = args.epochs
    config.model_name = args.model

    if config.device.lower() != "auto":
        print(f"Using GPU {config.device}")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    os.makedirs(config.log_folder, exist_ok=True)

    if args.mode == "bench":
        run_benchmark()
    elif args.mode == "train":
        train_model()
    else:
        raise Exception(f"Invalid mode {args.mode}")

if __name__ == "__main__":
    main()