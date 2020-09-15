"""
Rescue the General Environment


"""

import gym
import numpy as np
import itertools

# the initial health of each player
PLAYER_MAX_HEALTH = 10

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

class RescueTheGeneralEnv(MultiAgentEnv):
    """
    The rescue the general game.

    Rules:
    Players may occupy the same square.
    Red team and green team do not know the location of the general until they come within vision range

    Action Space:
    (see below)

    Observational Space
    """

    MAP_GRASS = 1
    MAP_TREE = 2

    TEAM_RED = 0
    TEAM_GREEN = 1
    TEAM_BLUE = 2

    ACTION_NOOP = 0
    ACTION_MOVE_UP = 1
    ACTION_MOVE_DOWN = 2
    ACTION_MOVE_LEFT = 3
    ACTION_MOVE_RIGHT = 4
    ACTION_SHOOT_UP = 5
    ACTION_SHOOT_DOWN = 6
    ACTION_SHOOT_LEFT = 7
    ACTION_SHOOT_RIGHT = 8
    ACTION_ACT = 9

    DX = [0, 0, -1, +1]
    DY = [-1, 1, 0, 0]


    def __init__(self):
        super().__init__()

        self.n_trees = 10
        self.map_width = 64
        self.map_height = 64
        self.n_players = 12
        self.map_layers = self.n_players + 1
        self.player_view_distance = 3

        # general location
        self.general_location = (np.random.randint(1, self.map_width - 1), np.random.randint(1, self.map_height - 1))

        # create map
        self.map = np.zeros((self.map_width, self.map_height, self.map_layers), dtype=np.uint8)
        all_locations = list(itertools.product(range(self.map_width), range(self.map_height)))
        for i in range(self.n_trees):
            tree_locations = np.random.choice(all_locations, self.n_trees, replace=False)
            for loc in tree_locations:
                self.map[loc] = 1

        # initialize players to random start locations, and assign initial health
        general_filter = lambda x,y: x-self.general_location[0] > 3 and y-self.general_location[1] > 3

        valid_start_locations = filter(general_filter, all_locations)
        start_locations = np.random.choice(valid_start_locations, self.n_players, replace=False)

        self.player_location = np.zeros((self.n_players, 2), dtype=np.uint8)
        self.player_health = np.zeros((self.n_players), dtype=np.uint8)
        self.player_seen_general = np.zeros((self.n_players), dtype=np.uint8)
        self.player_team = np.zeros((self.n_players), dtype=np.uint8)

        for i in range(self.n_players):
            self.player_location[i] = start_locations[i]
            self.player_health[i] = PLAYER_MAX_HEALTH

    def _in_vision(self, player_id, x, y):
        """
        Returns if given co-ordinates are within vision of the given player or not.
        """

        assert 0 <= player_id < self.n_players, "Invalid player_id"

        px, py = self.player_location[player_id]

        return abs(px - x) + abs(py - y) < self.player_view_distance

    def _move_player(self, player_id, dx, dy):
        """
        Moves player given offset.
        Players may occupy cells with other players, but if they move outside of the map they will be clipped back in.
        """

        assert 0 <= player_id < self.n_players, "Invalid player_id"

        x,y = self.player_location[player_id]
        x = np.clip(x + dx, 0, self.map_width)
        y = np.clip(y + dy, 0, self.map_height)
        self.player_location[player_id] = x,y

        # update general vision
        if self._in_vision(player_id, *self.general_location):
            self.player_seen_general[player_id] = True

    def step(self, actions):
        """
        Perform game step
        :param actions: np array of actions of dims [n_players]
        :return: observations, rewards, dones, infos
        """
        assert len(actions) == self.n_players, "Invalid number of players"

        rescue_counter = 0
        tree_counter = 0

        rewards = np.zeros([self.n_players], dtype=np.float)
        dones = np.zeros([self.n_players], dtype=np.bool)

        # apply actions, we process actions in the following order..
        # shooting
        # moving + acting

        for player_id in range(self.n_players):
            if player_id in range(self.n_players) in [self.ACTION_SHOOT_UP, self.ACTION_SHOOT_DOWN, self.ACTION_SHOOT_LEFT, self.ACTION_SHOOT_RIGHT]:
                # shooting is not implemented yet...
                pass

        for player_id in range(self.n_players):
            if actions[player_id] in [self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN, self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT]:
                indx = actions[player_id] - self.ACTION_MOVE_UP
                self._move_player(player_id, self.DX[indx], self.DY[indx])
            elif actions[player_id] == self.ACTION_ACT:
                if self.general_location == self.player_location[player_id]:
                    rescue_counter += 1
                if self.map[self.player_location[player_id]] == self.MAP_TREE:
                    tree_counter += 1
                    self.map[self.player_location[player_id]] = self.MAP_GRASS

        # generate points
        general_killed = False
        general_rescued = rescue_counter >= 2

        for player_id in range(self.n_players):
            if self.player_team == self.TEAM_GREEN:
                rewards[player_id] += tree_counter
            if general_killed:
                if self.player_team == self.TEAM_RED:
                    rewards[player_id] += 10
                elif self.player_team == self.TEAM_BLUE:
                    rewards[player_id] -= 10
            elif general_rescued:
                if self.player_team == self.TEAM_BLUE:
                    rewards[player_id] += 10
                elif self.player_team == self.TEAM_RED:
                    rewards[player_id] -= 10

        # end conditions

        # send done notifications to players who are dead
        for player_id in range(self.n_players):
            dones[player_id] = self.player_health <= 0

        # game ends for everyone under these outcome decisions
        if general_killed or general_rescued:
            dones[:] = True

        obs = [self._get_player_observation(player_id) for player_id in range(self.n_players)]
        infos = [{} for _ in range(self.n_players)]

        return obs, rewards, dones, infos



    def _get_player_observation(self, player_id):
        """
        Returns the player observation. Non-visible parts of map will be masked with 0.

        :param player_id:
        :return: Numpy array containing player observation of dims [map_layers, width, height],
            first layer is map
            next layer is general position (if known)
            then n_players layers with [0..1] at players location (if visible) indicating health

        """

        assert 0 <= player_id < self.n_players, "Invalid player_id"

        x, y = self.player_location[player_id]

        obs = np.zeros((self.map_layers, self.map_width, self.map_height), dtype=np.uint8)
        obs[0, :, :] = self.map

        if self.player_seen_general[player_id]:
            gx, gy = self.general_location
            obs[1, gx, gy] = 1

        for i in range(self.n_players):
            px, py = self.player_location[i]
            if self._in_vision(player_id, px, py):
                obs[i+2, px, py] = self.player_health / PLAYER_MAX_HEALTH

        return obs

    # Vectorized wrapper for a batch of multi-agent environments
# All environments must have the same observation and action space
class VectorizedMultiAgentEnv(gym.Env):
    def __init__(self):
        pass

