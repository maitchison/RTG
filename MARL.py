import gym
import numpy as np

from stable_baselines.common.vec_env.base_vec_env import VecEnv


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

    def step(self, actions):
        return [], [], [], []

    def reset(self):
        return []

class MultiAgentVecEnv(VecEnv):
    """
    Vectorized Adapter for multi-agent environments.

    This allows multi-agent environments to be played by single-agent algorithms.
    All players are played by the same model, but using different instances (if LSTM is used).

    """

    def __init__(self, make_marl_envs):
        """
        :param make_env: List of functions to make given environment
        """

        self.envs = [make_env() for make_env in make_marl_envs]
        self.num_envs = sum(env.n_players for env in self.envs)

        env = self.envs[0]
        VecEnv.__init__(self, self.num_envs, env.observation_space, env.action_space)

        self.actions = None
        self.auto_reset = True

    @property
    def n_players(self):
        return self.envs[0].n_players

    def get_roles(self):
        """
        Returns numpy array containing roles for each player
        :return: tensor of dims [n_envs, n_players]
        """
        roles = []
        for env in self.envs:
            env_roles = []
            for player in env.players:
                env_roles.append(player.team)
            for _ in env.players:
                roles.append(env_roles)
        return np.asarray(roles, dtype=np.int)


    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):

        obs = []
        rewards = []
        dones = []
        infos = []

        # step each marl environment
        reversed_actions = list(reversed(self.actions))
        for env in self.envs:
            env_actions = []
            for _ in range(env.n_players):
                env_actions.append(reversed_actions.pop())

            env_obs, env_rewards, env_dones, env_infos = env.step(env_actions)

            # auto reset.
            if self.auto_reset and all(env_dones):
                # save final terminal observation for later
                for this_info, this_obs in zip(env_infos, env_obs):
                    this_info['terminal_observation'] = this_obs
                    this_info['team_scores'] = env.team_scores.copy()
                env_obs = env.reset()

            obs.extend(env_obs)
            rewards.extend(env_rewards)
            dones.extend(env_dones)
            infos.extend(env_infos)

        # convert to np arrays
        obs = np.asarray(obs)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        return obs, rewards, dones, infos


    def seed(self, seed=None):
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        obs = []
        for env in self.envs:
            obs.extend(env.reset())
        return np.asarray(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]