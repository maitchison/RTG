"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # errors only
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from rescue import RescueTheGeneralEnv
from MARL import MultiAgentVecEnv

import numpy as np
import cv2
import os
import argparse
import time
import shutil

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
    cpus = 1 if tf.test.is_gpu_available else None  # for GPU machines its best to use only 1 CPU core
    parallel_envs = 64
    n_steps = 40

def export_video(filename, model, vec_env):
    """
    Exports a movie with agents playing randomly.
    """

    scale = 8

    states = vec_env.reset()
    env = vec_env.envs[0]
    frame = env.render("rgb_array")

    # work out our height
    height, width, channels = frame.shape
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height), isColor=True)

    dones = [False] * len(states)

    # don't like it that this is hard coded... not sure how to init the states?
    agent_states = model.initial_state

    n_players = env.n_players
    is_terminal = False
    outcome = ""

    # play the game...
    while not is_terminal:

        stacked_states = np.asarray(states)
        actions, _, agent_states, _ = model.step(stacked_states, agent_states, dones)

        states, rewards, dones, infos = vec_env.step(actions)

        # terminal state needs to be handled a bit differently as state will now be first state of new game
        # this is due to auto-reset.
        is_terminal = "terminal_rgb" in infos[0]

        # generate frames from global perspective
        if is_terminal:
            frame = infos[0]["terminal_rgb"]
            outcome = infos[0]["outcome"]
        else:
            frame = env.render("rgb_array")

        # for some reason cv2 wants BGR instead of RGB
        frame[:, :, :] = frame[:, :, ::-1]

        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert \
            frame.shape[1] == width and frame.shape[0] == height, \
            "Frame should be {} but is {}".format((width, height, 3), frame.shape)

        video_out.write(frame)

    video_out.release()

    # rename video to include outcome
    try:
        modified_filename = f"{os.path.splitext(filename)[0]} [{outcome}].{os.path.splitext(filename)[1]}"
        shutil.move(filename, modified_filename)
    except:
        print("Warning: could not rename video file.")


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
    vec_env = MultiAgentVecEnv([lambda: RescueTheGeneralEnv(scenario=config.scenario) for _ in range(config.parallel_envs)])
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
    export_video(f"{config.log_folder}/ppo_run_{config.epochs:03}_M.mp4", model, vec_env)

    # flush the log buffer
    for env in vec_env.envs:
        env.write_log_buffer()

    print("Finished training.")

def make_model(env, model_name = None):

    model_name = model_name or config.model_name

    model_func = lambda x: PPO2(
        CnnLstmPolicy,
        env,
        verbose=1,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        n_steps=config.n_steps,
        n_cpu_tf_sess=config.cpus,    # limiting cpu count really helps performance a lot when using GPU
        policy_kwargs=x
    )

    if model_name == "cnn_lstm_default":
        return model_func({
            "cnn_extractor": cnn_default,
            "n_lstm": 128
        })

    elif model_name == "cnn_lstm_fast":
        return model_func({
            "cnn_extractor": cnn_fast,
            "n_lstm": 64
        })
    else:
        raise ValueError(f"Invalid model name {model_name}")



def run_benchmark():

    def bench_scenario(scenario_name):

        vec_env = MultiAgentVecEnv([lambda: RescueTheGeneralEnv(scenario_name) for _ in range(config.parallel_envs)])

        _ = vec_env.reset()
        steps = 0

        start_time = time.time()

        while time.time() - start_time < 10:
            random_actions = np.random.randint(0, 10, size=[vec_env.num_envs])
            states, _, _, _ = vec_env.step(random_actions)
            steps += config.parallel_envs

        time_taken = (time.time() - start_time)

        print(f" - scenario {scenario_name} runs at {steps/time_taken:.0f} FPS.")

    def bench_model(model_name):

        vec_env = MultiAgentVecEnv([RescueTheGeneralEnv for _ in range(config.parallel_envs)])
        model = make_model(vec_env, model_name)

        states = np.asarray(vec_env.reset())

        model_states = model.initial_state
        model_masks = np.zeros((vec_env.num_envs,), dtype=np.uint8)
        steps = 0

        start_time = time.time()

        while time.time() - start_time < 10:

            actions, _, model_states, _ = model.step(states, model_states, model_masks)

            steps += config.parallel_envs

        time_taken = (time.time() - start_time)

        print(f" - model {model_name} runs at {steps / time_taken:.0f} FPS.")

    print("Benchmarking environments...")
    for scenario_name in ["default", "red2"]:
        bench_scenario(scenario_name)

    print("Benchmarking models...")
    for model_name in ["cnn_lstm_default", "cnn_lstm_fast"]:
        bench_model(model_name)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="[bench|train]")
    parser.add_argument('--run', type=str, help="run folder", default="test")
    parser.add_argument('--device', type=str, help="[0|1|2|3|AUTO]", default="auto")
    parser.add_argument('--scenario', type=str, help="[full|red2]", default="full")
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