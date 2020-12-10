import gym
import numpy as np

import multiprocessing

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

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

    def __init__(self, n_players):
        super().__init__()
        self._n_players = n_players

    def step(self, actions):
        return [], [], [], []

    def reset(self):
        return []

    @property
    def n_agents(self):
        return self.n_players

    @property
    def n_players(self):
        return self._n_players

class DummyMARLEnv(MultiAgentEnv):
    """
    Dummy environment with given number of actors.
    """

    def __init__(self, action_space, observation_space, n_players):
        super().__init__(n_players)
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        obs = np.zeros((self.n_players, *self.observation_space.shape), dtype=self.observation_space.dtype)
        rewards = np.zeros([self.n_players], dtype=np.float)
        dones = np.zeros([self.n_players], dtype=np.bool)
        infos = [{"train_mask":0} for _ in range(self.n_players)]
        return obs, rewards, dones, infos

    def reset(self):
        obs = np.zeros((self.n_players, *self.observation_space.shape), dtype=self.observation_space.dtype)
        return obs


class MultiAgentVecEnv(VecEnv):
    """
    Vectorized Adapter for multi-agent environments.

    This allows multi-agent environments to be played by single-agent algorithms.
    All players are played by the same model, but using different instances (if LSTM is used).

    Definitions

    n_players: The total number of players in the environment
    n_agents: The total number of (non scripted) players in the environment

    The Vector Environment surfaces observations only for the agents, for example the environment

    | Game 1     | Game 2            | Game 3 |
    | p1,     p2 | p1,    p2,     p3 | p1, p2 |
    | AI,     AI | AI,    script, AI | AI, AI |

    Would surface as a vector environment of length 6 (the scripted player masked out)

    This environment has 3 games, total_agents=7 and total_agents=6

    For compatibility num_envs is set to total_agents

    """

    def __init__(self, make_marl_envs):
        """
        :param make_env: List of functions to make given environment
        """

        self.games = [make_env() for make_env in make_marl_envs]

        VecEnv.__init__(self, self.total_agents, self.games[0].observation_space, self.games[0].action_space)

        self.actions = None
        self.auto_reset = True
        self.run_once = False # if true environment will pause after first reset

        self.env_completed = [False] * self.num_envs

        

    @property
    def total_players(self):
        return sum(game.n_players for game in self.games)

    @property
    def total_agents(self):
        return sum(game.n_agents for game in self.games)

    @property
    def max_players(self):
        return max(env.n_players for env in self.games)

    @property
    def max_roles(self):
        """ Returns number of roles in game"""
        # note, we take max here, so if a game has team 0 and team 2, 3 will be returned, with 0 players being on team 1
        result = 0
        for env in self.games:
            for player in env.players:
                result = max(result, 1 + player.team)
        return result

    def get_roles(self):
        """
        Returns numpy array containing roles for each player
        :return: tensor of dims [n_players]
        """
        roles = []
        for game in self.games:
            for player in game.players:
                roles.append(player.team)
        return np.asarray(roles, dtype=np.int64)


    def get_roles_expanded(self):
        """
        Returns numpy array containing roles for each player in game for each agent in environment ordered by
        public_id.
        :return: tensor of dims [n_agents, n_players]
        """

        # note, this might not work with scripted environments?
        roles = []
        for game in self.games:
            game_players = [(player.public_id, player.team) for player in game.players]
            game_players.sort()
            game_players = [team for id, team in game_players]

            for _ in game.players:
                roles.append(game_players)

        return np.asarray(roles, dtype=np.int64)

    def get_alive(self):
        """
        Returns numpy array containing alive status for each player
        :return: bool np array of dims [n_envs]
        """
        alive = []
        for game in self.games:
            for player in game.players:
                alive.append(player.is_alive)
        return np.asarray(alive, dtype=np.bool)

    def step_async(self, actions):
        actions = list(actions)
        assert len(actions) == self.num_envs,\
            f"Wrong number of actions, expected {self.num_envs} but found {len(actions)}."
        self.actions = actions

    def step_wait(self):

        obs = []
        rewards = []
        dones = []
        infos = []

        # step each marl environment
        reversed_actions = list(reversed(self.actions))
        for i, game in enumerate(self.games):

            if self.run_once and self.env_completed[i]:
                # ignore completed environments
                for _ in range(game.n_players):
                    reversed_actions.pop()
                blank_obs = np.zeros(game.observation_space.shape, dtype=game.observation_space.dtype)
                obs.extend([blank_obs]*game.n_players)
                rewards.extend([0]*game.n_players)
                dones.extend([True]*game.n_players)
                infos.extend([{}]*game.n_players)
                continue

            env_actions = []
            for _ in range(game.n_players):
                env_actions.append(reversed_actions.pop())

            env_obs, env_rewards, env_dones, env_infos = game.step(env_actions)

            # auto reset.
            if self.auto_reset and all(env_dones):
                # save final terminal observation for later
                for this_info, this_obs in zip(env_infos, env_obs):
                    this_info['terminal_observation'] = this_obs
                    this_info['team_scores'] = game.team_scores.copy()
                if not self.run_once:
                    env_obs = game.reset()
                self.env_completed[i] = True

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
        for idx, game in enumerate(self.games):
            seeds.append(game.seed(seed + idx))
        return seeds

    def reset(self):
        obs = []
        for game in self.games:
            obs.extend(game.reset())
        return np.asarray(obs)

    def close(self):
        for game in self.games:
            game.close()

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
        return [self.games[i] for i in indices]