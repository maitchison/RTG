"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

from rescue import RescueTheGeneralEnv
import numpy as np

def main():

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



if __name__ == "__main__":
    main()