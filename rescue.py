"""
Rescue the General Environment


"""

import gym
import numpy as np

class MultiAgentEnv(gym.Env):
    """
    Multi-Agent extension for OpenAI Gym

    Credit:
    Largely based on OpenAI's multi-agent environment
    https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/environment.py

    """
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self):
        super().__init__()
        pass

    def step(self, action_n):
        pass

    def reset(self):
        pass


    def _get_obs(self, agent):
        """ get observation for a particular agent """
        pass

    def _get_reward(self, agent):
        """ get reward for a particular agent """
        pass

    def _set_action(self, action, agent, action_space):
        """ set env action for a particular agent """
        pass

    def render(self, mode='human'):
        """ render environment """
        pass

class MultAgentMapEnv(MultiAgentEnv):
    """
    Defines a map-based multi-agent environment. Agents may have global or egocentric views.
    """

    def __init__(self, egocentric=True, vision="ergo"):
        """
        :param egocentric:
        :param vision: ["global", "partial", "ergo"]
        """
        super().__init__()
        self.egocentric = egocentric
        self.vision = vision


    def _get_obs(self, agent):
        """ get observation for a particular agent """

        if self.vision == "global":
            pass
        elif self.vision == "partial":
            pass
        elif self.vision == "ergo":
            pass
        else:
            raise ValueError(f"Invalid vision type {self.vision}")

class RescueTheGeneralEnv(MultAgentMapEnv):
    """
    The rescue the general game.
    """

    def __init__(self):
        super().__init__()

# Vectorized wrapper for a batch of multi-agent environments
# All environments must have the same observation and action space
class VectorizedMultiAgentEnv(gym.Env):
    def __init__(self):
        pass

