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

    def step(self, actions):
        pass

    def reset(self):
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
    Rescuing and killing the general simultaneously counts as the general being killed.

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
        self.map_layers = self.n_players + 2
        self.player_view_distance = 3
        self.timeout = 1000
        self.counter = 0

        self.player_location = np.zeros((self.n_players, 2), dtype=np.uint8)
        self.player_health = np.zeros((self.n_players), dtype=np.uint8)
        self.player_seen_general = np.zeros((self.n_players), dtype=np.uint8)
        self.player_team = np.zeros((self.n_players), dtype=np.uint8)

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
        x = np.clip(x + dx, 0, self.map_width-1)
        y = np.clip(y + dy, 0, self.map_height-1)
        self.player_location[player_id] = x, y

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

        # dead men can't act
        for i in range(self.n_players):
            if self.player_health[i] <= 0:
                actions[i] = self.ACTION_NOOP

        # apply actions, we process actions in the following order..
        # shooting
        # moving + acting

        for i in range(self.n_players):
            if i in range(self.n_players) in [self.ACTION_SHOOT_UP, self.ACTION_SHOOT_DOWN, self.ACTION_SHOOT_LEFT, self.ACTION_SHOOT_RIGHT]:
                # shooting is not implemented yet...
                pass

        for i in range(self.n_players):
            if actions[i] in [self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN, self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT]:
                indx = actions[i] - self.ACTION_MOVE_UP
                self._move_player(i, self.DX[indx], self.DY[indx])
            elif actions[i] == self.ACTION_ACT:
                px, py = self.player_location[i]
                if self.general_location == (px, py):
                    rescue_counter += 1
                if self.map[(px, py)] == self.MAP_TREE:
                    tree_counter += 1
                    self.map[(px, py)] = self.MAP_GRASS

        # generate points
        general_killed = False
        general_rescued = rescue_counter >= 2 and not general_killed
        game_timeout = self.counter >= self.timeout

        for i in range(self.n_players):
            if self.player_team[i] == self.TEAM_GREEN:
                rewards[i] += tree_counter
            if general_killed:
                if self.player_team[i] == self.TEAM_RED:
                    rewards[i] += 10
                elif self.player_team[i] == self.TEAM_BLUE:
                    rewards[i] -= 10
            elif general_rescued:
                if self.player_team[i] == self.TEAM_BLUE:
                    rewards[i] += 10
                elif self.player_team[i]== self.TEAM_RED:
                    rewards[i] -= 10
            elif game_timeout:
                if self.player_team[i] == self.TEAM_GREEN:
                    rewards[i] += 10

        # send done notifications to players who are dead
        for i in range(self.n_players):
            dones[i] = self.player_health[i] <= 0

        # game ends for everyone under these outcome decisions
        if general_killed or general_rescued or game_timeout:
            dones[:] = True

        obs = self._get_observations()
        infos = [{} for _ in range(self.n_players)]

        self.counter += 1

        return obs, rewards, dones, infos

    def _get_observations(self):
        return [self._get_player_observation(player_id) for player_id in range(self.n_players)]

    def _get_player_observation(self, player_id):
        """
        Returns the player observation. Non-visible parts of map will be masked with 0.

        :param player_id:
        :return: Numpy array containing player observation of dims [map_layers, width, height],
            first layer is map
            next layer is general position (if known)
            then n_players layers with number at players location (if visible) indicating health
        """

        assert 0 <= player_id < self.n_players, "Invalid player_id"

        obs = np.zeros((self.map_layers, self.map_width, self.map_height), dtype=np.uint8)
        obs[0, :, :] = self.map[:, :]

        if self.player_seen_general[player_id]:
            gx, gy = self.general_location
            obs[1, gx, gy] = 1

        for i in range(self.n_players):
            px, py = self.player_location[i]
            if self._in_vision(player_id, px, py):
                obs[i+2, px, py] = self.player_health[i]

        return obs

    def reset(self):
        """
        Reset game.
        :return: observations
        """

        # general location
        self.general_location = (np.random.randint(1, self.map_width - 2), np.random.randint(1, self.map_height - 2))

        # create map
        self.map = np.zeros((self.map_width, self.map_height), dtype=np.uint8)
        all_locations = list(itertools.product(range(self.map_width), range(self.map_height)))
        for i in range(self.n_trees):
            idxs = np.random.choice(len(all_locations), self.n_trees, replace=False)
            for loc in [all_locations[idx] for idx in idxs]:
                self.map[loc] = 1

        # initialize players to random start locations, and assign initial health
        general_filter = lambda p: p[0] - self.general_location[0] > 3 and p[1] - self.general_location[1] > 3

        valid_start_locations = list(filter(general_filter, all_locations))
        start_locations = [valid_start_locations[idx] for idx in np.random.choice(len(valid_start_locations), self.n_players, replace=False)]

        self.player_location *= 0
        self.player_health *= 0
        self.player_seen_general *= 0
        self.player_team *= 0

        # for the moment hardcode 4,4,4 teams
        assert self.n_players == 12
        teams = [self.TEAM_RED]*4 + [self.TEAM_GREEN]*4 + [self.TEAM_BLUE]*4
        np.random.shuffle(teams)
        self.player_team[:] = teams

        for i in range(self.n_players):
            self.player_location[i] = start_locations[i]
            self.player_health[i] = PLAYER_MAX_HEALTH
            self.player_seen_general[i] = self.player_team[i] == self.TEAM_BLUE
            # this will update seen_general if player is close
            # which normally doesn't happen as I make sure players do not start close to the general
            self._move_player(i, 0, 0)

        return self._get_observations()

    def render(self, mode='human'):
        pass

