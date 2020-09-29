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
from tools import load_results, get_score, get_score_alt

import uuid
import numpy as np
import cv2
import os
import argparse
import time
import shutil
from ast import literal_eval

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines import PPO2

from new_models import cnn_default, cnn_fast

class Config():
    """ Class to hold config files"""

    def __init__(self):
        self.log_folder = str()
        self.device = str()
        self.scenario = str()
        self.epochs = int()
        self.model_name = str()
        self.cpus = 1 if tf.test.is_gpu_available else None  # for GPU machines its best to use only 1 CPU core
        self.parallel_envs = int()
        self.algo_params = dict()

    def __str__(self):
        return str(dict(vars(self).items()))

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
        modified_filename = f"{os.path.splitext(filename)[0]} [{outcome}]{os.path.splitext(filename)[1]}"
        shutil.move(filename, modified_filename)
    except:
        print("Warning: could not rename video file.")


def train_model():
    """
    Train PPO on the environment using the "other agents are environment" method.
    :return:
    """

    print("="*60)

    # copy source files for later
    from shutil import copyfile
    for filename in ["train.py", "rescue.py"]:
        copyfile(filename, f"{config.log_folder}/{filename}")

    # make a copy of the environment parameters
    with open(f"{config.log_folder}/config.txt", "w") as f:
        f.write(str(config))

    # our MARL environments are handled like vectorized environments
    vec_env = MultiAgentVecEnv([lambda: RescueTheGeneralEnv(scenario=config.scenario) for _ in range(config.parallel_envs)])
    print("Scenario parameters:")
    print(vec_env.envs[0].scenario)

    model = make_model(vec_env)

    for sub_env in vec_env.envs:
        sub_env.log_folder = config.log_folder

    print("="*60)

    start_time = time.time()

    for epoch in range(0, config.epochs):

        export_video(f"{config.log_folder}/ppo_run_{epoch:03}_M.mp4", model, vec_env)
        model.save(f"{config.log_folder}/model_{epoch:03}_M.p")
        print()
        print(f"Training epoch {epoch} on experiment {config.log_folder}")
        print()

        for sub_epoch in range(10):
            start_epoch_time = time.time()
            model.learn(100000, reset_num_timesteps=False, log_interval=10)
            epoch_time = time.time() - start_epoch_time
            fps = 100000 / epoch_time
            if sub_epoch==0:
                print(f"FPS: {fps:.0f} .", end='', flush=True)
            else:
                print(".", end='', flush=True)
        print()

        # flush the log buffer and print scores
        for env in vec_env.envs:
            env.write_log_buffer()

        print_scores()

    model.save(f"{config.log_folder}/model_final.p")
    export_video(f"{config.log_folder}/ppo_run_{config.epochs:03}_M.mp4", model, vec_env)

    time_taken = time.time() - start_time
    print(f"Finished training after {time_taken/60/60:.1f}h.")

def make_model(env, model_name = None, verbose=0):

    model_name = model_name or config.model_name

    # figure out a mini_batch size
    batch_size = config.algo_params["n_steps"] * env.num_envs
    assert batch_size % config.algo_params["mini_batch_size"] == 0, \
        f"Invalid mini_batch size {config.algo_params['mini_batch_size']}, must divide batch size {batch_size}."
    n_mini_batches = batch_size // config.algo_params["mini_batch_size"]

    params = config.algo_params.copy()
    del params["mini_batch_size"]

    model_func = lambda x: PPO2(
        CnnLstmPolicy,
        env,
        verbose=verbose,
        nminibatches=n_mini_batches,
        policy_kwargs=x,
        **params
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
            steps += vec_env.num_envs

        time_taken = (time.time() - start_time)

        print(f" - scenario {scenario_name} runs at {steps / time_taken / 1000:.1f}k AAPS.")

    def bench_training(scenario_name, model_name):

        vec_env = MultiAgentVecEnv([lambda: RescueTheGeneralEnv(scenario_name) for _ in range(config.parallel_envs)])
        model = make_model(vec_env, model_name, verbose=0)

        # just to warm it up
        model.learn(10)

        start_time = time.time()
        model.learn(10000)
        time_taken = (time.time() - start_time)

        print(f" - model {model_name} trains at {10000 / time_taken / 1000:.1f}k FPS.")

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

            steps += vec_env.num_envs

        time_taken = (time.time() - start_time)

        print(f" - model {model_name} runs at {steps / time_taken / 1000:.1f}k AAPS.")

    print("Benchmarking training...")
    for model_name in ["cnn_lstm_default", "cnn_lstm_fast"]:
        bench_training("red2", model_name)

    print("Benchmarking environments...")
    for scenario_name in ["full", "medium", "red2"]:
        bench_scenario(scenario_name)

    print("Benchmarking models...")
    for model_name in ["cnn_lstm_default", "cnn_lstm_fast"]:
        bench_model(model_name)

def print_scores():
    """ Print current scores"""
    try:
        results = load_results(config.log_folder)
    except:
        # this just means results have not generated yet
        return

    print("Team scores:")
    for team in ("red", "green", "blue"):
        score = get_score(results, team)
        alt_score = get_score_alt(results, team)
        print(f"  {team:<8}: {score:<4.1f} {alt_score:<4.1f}")

def regression_test():

    print("Performing regression test, this could take some time.")

    start_time = time.time()

    def run_test(scenario_name, n_steps=int(2e6)):

        destination_folder = os.path.join(config.log_folder, scenario_name)
        os.makedirs(destination_folder, exist_ok=True)

        # our MARL environments are handled like vectorized environments
        vec_env = MultiAgentVecEnv(
            [lambda: RescueTheGeneralEnv(scenario=scenario_name) for _ in range(config.parallel_envs)])
        model = make_model(vec_env, verbose=0)

        for sub_env in vec_env.envs:
            sub_env.log_folder = destination_folder

        model.learn(n_steps, reset_num_timesteps=False)
        export_video(f"{destination_folder}/{scenario_name}.mp4", model, vec_env)

        # flush the log buffer
        for env in vec_env.envs:
            env.write_log_buffer()

        # check scores
        results = load_results(destination_folder)

        return results

    for scenario_name, team, required_score in [
        ("red2", "red", 7.5),
        ("green2", "green", 7.5),
        ("blue2", "blue", 7.5),
    ]:

        results = run_test(scenario_name)
        score = get_score_alt(results, team)

        result = "FAIL" if score < required_score else "PASS"
        print(f"  [{result:}] {scenario_name:<20} ({score:.1f})")

    time_taken = time.time() - start_time
    print(f"Finished tests in {time_taken/60:.1f}m.")


def main():

    DEFAULT_ALGO_PARAMS = {
        'learning_rate': 2.5e-4,
        'n_steps': 128,
        'ent_coef': 0.01,
        'n_cpu_tf_sess': 1,
        'mini_batch_size': 2048,
        'max_grad_norm': 2.0,
        'cliprange_vf': -1      # this has been shown to not be effective so I disable it
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="[bench|train|test]")
    parser.add_argument('--run', type=str, help="run folder", default="test")
    parser.add_argument('--device', type=str, help="[0|1|2|3|AUTO]", default="auto")
    parser.add_argument('--scenario', type=str, help="[full|red2]", default="full")
    parser.add_argument('--epochs', type=int, help="number of epochs to train for (each 1M agent steps)", default=100)
    parser.add_argument('--model', type=str, help="model to use [cnn_lstm_default|cnn_lstm_fast]", default="cnn_lstm_default")

    parser.add_argument('--algo_params', type=str, default="{}")
    parser.add_argument('--parallel_envs', type=int, default=32)

    args = parser.parse_args()

    # setup config
    config.device = args.device
    config.scenario = args.scenario
    config.epochs = args.epochs
    config.model_name = args.model
    config.parallel_envs = args.parallel_envs
    config.algo_params = DEFAULT_ALGO_PARAMS
    config.algo_params.update(literal_eval(args.algo_params))
    config.uuid = uuid.uuid4().hex[-8:]
    config.log_folder = f"run/{args.run} [{config.uuid}]"


    if config.device.lower() != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    print()
    print(f"Starting {config.log_folder} on device {config.device}")

    os.makedirs(config.log_folder, exist_ok=True)

    if args.mode == "bench":
        run_benchmark()
    elif args.mode == "train":
        train_model()
    elif args.mode == "test":
        regression_test()
    else:
        raise Exception(f"Invalid mode {args.mode}")

if __name__ == "__main__":
    config = Config()
    main()