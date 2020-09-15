"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

from rescue import RescueTheGeneralEnv
import numpy as np
import cv2

def export_movie(filename):

    """ Exports a movie with agents playing randomly.
        which_frames: model, real, or both
    """

    scale = 8

    env = RescueTheGeneralEnv()
    obs = env.reset()
    frame = env.render("rgb_array")

    # work out our height
    height, width, channels = frame.shape
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

    dones = [False] * env.n_players

    # play the game...
    while not np.all(dones):

        actions = np.random.randint(0, 10, size=[env.n_players])
        states, rewards, dones, infos = env.step(actions)

        frame = env.render("rgb_array")
        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format((width, height, 3), frame.shape)

        video_out.write(frame)

    video_out.release()

def play_test_game():

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

def main():
    print("Rendering movie...")
    export_movie("test.mp4")
    print("Done.")


if __name__ == "__main__":
    main()