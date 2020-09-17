"""
Rescue the General Environment


"""

import gym
from gym import spaces
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

    def __init__(self, n_players):
        super().__init__()
        self.n_players = n_players

    def step(self, actions):
        return [], [], [], []

    def reset(self):
        return []

class MultAgentEnvAdapter(gym.Env):
    """
    Adapter for multi-agent environments. Trains a single agent an a muti-agent environment where all other agents
    are played by froozen copies of the agent.

    This allows multi-agent environments to be played by single-agent algorithms

    """

    def __init__(self, env:MultiAgentEnv):
        """
        :param multi_env: The multi-agent environment to use.
        """
        super().__init__()
        self.env = env
        self.model = None
        self.next_actions = np.zeros([self.env.n_players], dtype=np.uint8)

        self.action_space = env.action_space
        self.observation_space = env.observation_space


    def step(self, action):

        assert self.model is not None, "Must assign model for other actors before using environment"

        actions = self.next_actions[:]
        actions[0] = action

        obs, rewards, dones, infos = self.env.step(actions)

        # todo: implement LSTM
        self.next_actions, _, _, _ = self.model.step(np.asarray(obs), None, None)

        return obs[0], rewards[0], dones[0], infos[0]

    def reset(self):
        return self.env.reset()[0]

    def render(self, mode='human'):
        return self.env.render(mode)



class RescueTheGeneralEnv(MultiAgentEnv):
    """
    The rescue the general game.

    Rules:
    * Players may occupy the same square (stacking).
    * Red team and green team do not know the location of the general until they come within vision range
    * Rescuing and killing the general simultaneously counts as the general being killed.
    * Players may shoot in any direction, but gun has limited range (usually less than vision)
    * If players are stacked and are hit by a bullet a random  player on tile is hit
    * If a player shoots but other players are stacked ontop then a random player at the shooters location is hit (other
       than the shooter themselves)
    * If game times out then any green players still alive are awared 5 points

    Action Space:
    (see below)

    Observational Space
    """

    MAP_GRASS = 1
    MAP_TREE = 2

    TEAM_RED = 0
    TEAM_GREEN = 1
    TEAM_BLUE = 2

    TEAM_COLOR = [
        (255,25,25),
        (25,255,25),
        (25,25,255)
    ]

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

    SHOOT_ACTIONS = [ACTION_SHOOT_UP, ACTION_SHOOT_DOWN, ACTION_SHOOT_LEFT, ACTION_SHOOT_RIGHT]

    DX = [0, 0, -1, +1]
    DY = [-1, 1, 0, 0]

    def __init__(self):
        super().__init__(n_players=12)

        self.n_trees = 10
        self.map_width = 48
        self.map_height = 48
        self.map_layers = self.n_players + 3
        self.player_view_distance = 5
        self.player_shoot_range = 12
        self.timeout = 1000
        self.counter = 0

        self.general_location = (0,0)
        self.general_health = 10

        self.player_location = np.zeros((self.n_players, 2), dtype=np.uint8)
        self.player_health = np.zeros((self.n_players), dtype=np.uint8)
        self.player_seen_general = np.zeros((self.n_players), dtype=np.uint8)
        self.player_team = np.zeros((self.n_players), dtype=np.uint8)
        self.player_last_action = np.zeros((self.n_players), dtype=np.uint8)

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.map_width, self.map_height, self.map_layers), dtype=np.uint8
        )

    def _in_vision(self, player_id, x, y):
        """
        Returns if given co-ordinates are within vision of the given player or not.
        """

        assert 0 <= player_id < self.n_players, "Invalid player_id"

        px, py = self.player_location[player_id]

        return abs(int(px) - x) + abs(int(py) - y) < self.player_view_distance

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
            if actions[i] in self.SHOOT_ACTIONS:
                indx = actions[i] - self.ACTION_SHOOT_UP
                x, y = self.player_location[i]
                target_hit = False
                for j in range(self.player_shoot_range):
                    # check location
                    for k in range(self.n_players):
                        if k == i:
                            continue
                        if (x,y) == self.general_location:
                            # general was hit
                            self.general_health -= (np.random.randint(1,6) + np.random.randint(1,6))
                        if (x,y) == tuple(self.player_location[k]) and self.player_health[k] > 0:
                            # this player got hit
                            # todo: randomize who gets hit if players are stacked
                            self._damage_player(k, np.random.randint(1,6) + np.random.randint(1,6))
                            target_hit = True

                    x += self.DX[indx]
                    y += self.DY[indx]

                    if target_hit:
                        break

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
        general_killed = self.general_health <= 0
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
                    if self.player_health[i] > 0:
                        rewards[i] += 5

        # send done notifications to players who are dead
        for i in range(self.n_players):
            dones[i] = self.player_health[i] <= 0

        # game ends for everyone under these outcome decisions
        if general_killed or general_rescued or game_timeout:
            dones[:] = True

        obs = self._get_observations()
        infos = [{} for _ in range(self.n_players)]

        self.counter += 1
        self.player_last_action = actions

        return obs, rewards, dones, infos

    def _damage_player(self, player_id, damage):
        """
        Causes player to receive given damage.
        :param player_id:
        :param damage:
        :return:
        """
        self.player_health[player_id] = np.clip(self.player_health[player_id] - damage, 0, PLAYER_MAX_HEALTH)

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

            # rotate so this player is always in the first spot.
            destination_layer = ((i + self.n_players - player_id) % self.n_players) + 2

            if self._in_vision(player_id, px, py):
                obs[destination_layer, px, py] = self.player_health[i]
                if self.player_last_action[i] in self.SHOOT_ACTIONS:
                    indx = self.player_last_action[i] - self.ACTION_SHOOT_UP
                    px += self.DX[indx]
                    py += self.DY[indx]
                    # not sure this is the best way to indicate a player shooting?
                    # maybe I should draw the marker on the spot hit...
                    if 0 <= px < self.map_width and 0 <= py < self.map_height:
                        obs[destination_layer, px, py] = 1

        # final layer is just our team, this would be better feed in as non-spatial data...
        obs[-1,:,:] = self.player_team[player_id]

        # I really don't like this, we are shifting from CWH to WHC, which is slow, and pytorch
        # should expect channels first, but for some reason baselines wants them last
        # oh.. wait.. yeah this is in tensorflow... hmm ok.
        obs = obs.swapaxes(0, 2)
        obs = obs.swapaxes(0, 1)

        return obs

    def reset(self):
        """
        Reset game.
        :return: observations
        """

        # general location
        self.general_location = (np.random.randint(1, self.map_width - 2), np.random.randint(1, self.map_height - 2))
        self.general_health = 10

        # create map
        self.map = np.zeros((self.map_width, self.map_height), dtype=np.uint8) + 1

        all_locations = list(itertools.product(range(self.map_width), range(self.map_height)))
        idxs = np.random.choice(len(all_locations), size=self.n_trees, replace=False)
        for loc in [all_locations[idx] for idx in idxs]:
            self.map[loc] = 2

        # initialize players to random start locations, and assign initial health
        general_filter = lambda p: abs(p[0] - self.general_location[0]) > 3 and abs(p[1] - self.general_location[1]) > 3

        valid_start_locations = list(filter(general_filter, all_locations))
        start_locations = [valid_start_locations[idx] for idx in np.random.choice(len(valid_start_locations), size=self.n_players, replace=False)]

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

    def _render_human(self):
        raise NotImplemented("Sorry tile-map rendering not implemented yet")

    def _render_rgb(self):
        """ Renders game in RGB using very simple colored tiles."""

        image = np.zeros((self.map_width, self.map_height, 3), dtype=np.uint8)

        for x in range(self.map_width):
            for y in range(self.map_height):

                c = [0,0,0]

                # first get color from terrain
                if self.map[x,y] == self.MAP_GRASS:
                    c = (150,75,0)
                elif self.map[x, y] == self.MAP_TREE:
                    c = (125,255,150)

                # general location
                if self.general_location == (x, y):
                    c = (25,255,255)

                # next get players color.. this is not ideal if they overlap
                for i in range(self.n_players):
                    if tuple(self.player_location[i]) == (x, y):
                        if self.player_health[i] > 0:
                            c = self.TEAM_COLOR[self.player_team[i]]
                        else:
                            # dead body
                            c = (0, 0, 0)

                # bullet flares
                for i in range(self.n_players):
                    if self.player_last_action[i] in self.SHOOT_ACTIONS:
                        indx = self.player_last_action[i] - self.ACTION_SHOOT_UP
                        fx, fy = self.player_location[i]
                        fx += self.DX[indx]
                        fy += self.DY[indx]
                        if (fx, fy) == (x,y):
                            c = (255,255,128)

                image[x,y,::-1] = c

        return image

    def render(self, mode='human'):
        """ render environment """
        if mode == 'human':
            return self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb()
        else:
            raise ValueError(f"Invalid render mode {mode}")

