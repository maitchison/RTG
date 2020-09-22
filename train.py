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

from rescue import RescueTheGeneralEnv, MultiAgentEnvAdapter

import numpy as np
import cv2
import os
import pickle
import argparse
import time

from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.policies import CnnLstmPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines import PPO2

from new_models import single_layer_cnn

class config():

    log_folder = str()



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
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height), isColor=True)

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
        video_out.write(frame)

    for _ in range(4):
        # this just makes it easier to see the last frame
        video_out.write(frame)

    video_out.release()

    return team_scores

def make_env():
    env = RescueTheGeneralEnv()
    env = MultiAgentEnvAdapter(env)
    return env

def train_model():
    """
    Train PPO on the environment using the "other agents are environment" method.
    :return:
    """

    print("Starting environment")

    # mutli-processor not supported yet. Would require sending the model to each process, and I don't know if
    # tensorflow allows instances running across processes like that.
    vec_env = DummyVecEnv([make_env for _ in range(16)])

    # create model
    model = PPO2(CnnPolicy, vec_env, verbose=1, learning_rate=2.5e-4, ent_coef=0.001, policy_kwargs={"cnn_extractor":single_layer_cnn})

    for sub_env in vec_env.envs:
        sub_env.model = model
        sub_env.env.log_folder = config.log_folder

    scores = []

    scores.append(export_movie(f"{config.log_folder}/ppo_run-0.mp4", model))

    # this is mostly just to see how big the model is..
    model.save(f"{config.log_folder}/model_initial.p")

    for epoch in range(1000):

        model.learn(100000, reset_num_timesteps=epoch==0, log_interval=10)

        print("Generating movie...")
        scores.append(export_movie(f"{config.log_folder}/ppo_run-{epoch+1}.mp4", model))

        pickle.dump(scores, open(f"{config.log_folder}/results.dat", "wb"))

    print("Finished training.")

def generate_video():
    """
    Generate a video of random play then exit.
    """

    print("Generating video.")
    model = PPO2(CnnPolicy, None, verbose=1, learning_rate=2.5e-4, ent_coef=0.001, policy_kwargs={"cnn_extractor":single_layer_cnn})
    export_movie(f"{config.log_folder}/ppo_run-0.mp4", model)

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
    parser.add_argument('--experiment', type=str, help="experiment folder", default="test")

    args = parser.parse_args()

    # setup config
    config.log_folder = args.experiment
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