"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""


from rescue import RescueTheGeneralEnv, MultAgentEnvAdapter

import numpy as np
import cv2
import pickle

import gym

from stable_baselines.common.policies import CnnLstmPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

def export_movie(filename, model):

    """ Exports a movie with agents playing randomly.
        which_frames: model, real, or both
    """

    scale = 8

    env = RescueTheGeneralEnv()
    states = env.reset()
    frame = env.render("rgb_array")

    # work out our height
    height, width, channels = frame.shape
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

    dones = [False] * env.n_players

    team_scores = np.zeros([3], dtype=np.int)

    # play the game...
    while not np.all(dones):

        # todo: implement LSTM
        actions, _, _, _ = model.step(np.asarray(states), None, None)

        states, rewards, dones, infos = env.step(actions)

        for i in range(env.n_players):
            team_scores[env.player_team[i]] += rewards[i]

        if np.sum(np.asarray(rewards)) != 0:
            print(rewards)

        frame = env.render("rgb_array")
        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format((width, height, 3), frame.shape)

        video_out.write(frame)

    video_out.release()

    return team_scores

def play_random_game():
    """
    Simple test of the environment using random actions...
    """

    print("Starting environment")

    env = RescueTheGeneralEnv()
    obs = env.reset()

    while True:
        print(f"Turn {env.counter}")
        # random actions for the moment...
        actions = np.random.randint(0, 10, size=[env.n_players])
        obs, rewards, dones, infos = env.step(actions)
        if np.all(dones):
            break


def play_simple_game():
    """
    Train PPO on the environment using the "other agents are environment" method.
    :return:
    """

    print("Starting environment")


    env = RescueTheGeneralEnv()
    env = MultAgentEnvAdapter(env)

    # create model
    model = PPO2(CnnPolicy, env, verbose=1)

    env.model = model

    scores = []

    scores.append(export_movie(f"ppo_run-initial.mp4", model))
    pickle.dump(scores, open("results.dat", "wb"))

    for epoch in range(100):

        model.learn(100000)

        print("Finished training.")
        model.save(f"model-{epoch}")
        print("Generating movie...")
        scores.append(export_movie(f"ppo_run-{epoch}.mp4", model))

        pickle.dump(scores, open("results.dat", "wb"))

def main():
    print("Rendering movie...")
    play_simple_game()
    print("Done.")


if __name__ == "__main__":
    main()