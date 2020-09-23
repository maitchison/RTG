#cython: language_level=3

"""
Rescue the General Environment


Suggested scenarios

************************
**Red-team-only**
************************

1 red players
0 green players
0 blue players

mode=easy (this won't matter much though
general=visible to all

Description:
This is a very simple scenario that should be solvable using standard single-agent algorithms.
It provides a good sanity check for learning algorithms, which should be able to solve the environment (average
score >9.9) in about 1M agent steps.

"""

"""
Benchmarking
Start, 1591
More efficent cropping, 3020  
"""

import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
import math
import gym.spaces

from stable_baselines.common.misc_util import mpi_rank_or_zero

from MARL import MultiAgentEnv

# the initial health of each player
PLAYER_MAX_HEALTH = 10

CELL_SIZE = 3
SIN_CHANNELS = 10 # 10 channels gets a frequence of 2^5, which suits maps around 32 tiles in length

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


    ---- new rules

    * players can not move onto trees, or the general
    * act now selects an adjacent object of interest, tree or general, automatically
    * if two of more players try to occupy the same square none of them move
    * unsure about blocking at the moment, I think I'll allow players to pass through.


    Action Space:
    (see below)

    Observational Space
    """

    REWARD_SCALE = 1 # some algorithms work better if value is higher

    MAP_GRASS = 1
    MAP_TREE = 2

    TEAM_RED = 0
    TEAM_GREEN = 1
    TEAM_BLUE = 2

    COLOR_FIRE = np.asarray([255, 255, 0], dtype=np.uint8)
    COLOR_HIGHLIGHT = np.asarray([180, 180, 50], dtype=np.uint8)
    COLOR_GENERAL = np.asarray([255, 255, 255], dtype=np.uint8)
    COLOR_NEUTRAL = np.asarray([64, 64, 64], dtype=np.uint8)
    COLOR_GRASS = np.asarray([0, 128, 0], dtype=np.uint8)
    COLOR_TREE = np.asarray([125, 255, 150], dtype=np.uint8)
    COLOR_DEAD = np.asarray([0, 0, 0], dtype=np.uint8)

    PLAYER_COLOR = np.asarray(
        [np.asarray(plt.cm.tab20(i)[:3])*255 for i in range(12)]
    , dtype=np.uint8)

    TEAM_COLOR = np.asarray([
        (255,25,25),
        (25,255,25),
        (25,25,255)
    ], dtype=np.uint8)

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
        super().__init__()

        self.log_folder = "./"

        self.n_players_red, self.n_players_green, self.n_players_blue = 2, 2, 2

        self.id = mpi_rank_or_zero()
        self.counter = 0
        self.game_counter = 0

        self._log_buffer = []

        self.general_location = (0,0)
        self.general_health = 0

        self.player_location = np.zeros((self.n_players, 2), dtype=np.int)
        self.player_health = np.zeros((self.n_players), dtype=np.int)
        self.player_seen_general = np.zeros((self.n_players), dtype=np.uint8)
        self.player_team = np.zeros((self.n_players), dtype=np.int)
        self.player_last_action = np.zeros((self.n_players), dtype=np.int)

        self.team_scores = np.zeros([3], dtype=np.int)

        # game rules
        self.easy_rewards = True # enables some easy rewards, such as killing enemy players.
        self.n_trees = 20
        self.map_width = 32
        self.map_height = 32
        self.player_view_distance = 5
        self.player_shoot_range = 4
        self.timeout = 1000
        self.general_always_visible = False
        self.initial_general_health = 10
        self.hidden_roles = False

        # other rules
        self.location_encoding = "abs" # none | sin | abs

        self.reward_logging = True # logs rewards to txt file

        self.stats_player_hit = np.zeros((3,3), dtype=np.int) # which teams killed who
        self.stats_deaths = np.zeros((3), dtype=np.int)  # how many players died
        self.stats_kills = np.zeros((3), dtype=np.int)  # how many players died
        self.stats_general_shot = np.zeros((3), dtype=np.int)  # which teams shot general
        self.stats_tree_harvested = np.zeros((3), dtype=np.int)  # which teams harvested trees
        self.stats_shots_fired = np.zeros((3), dtype=np.int)  # how many times each team shot
        self.stats_times_moved = np.zeros((3), dtype=np.int)  # how many times each team moved
        self.stats_times_acted = np.zeros((3), dtype=np.int)  # how many times each team acted
        self.stats_actions = np.zeros((3), dtype=np.int)  # how actions this team could have performed (sum_t(agents_alive))

        self.action_space = gym.spaces.Discrete(10)

        obs_channels = 3
        if self.location_encoding == "none":
            pass
        elif self.location_encoding == "sin":
            obs_channels += SIN_CHANNELS
        elif self.location_encoding == "abs":
            obs_channels += 2
        else:
            raise Exception(f"Invalid location encoding {self.location_encoding} use [none|sin|abs].")

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=((self.player_view_distance * 2 + 3) * CELL_SIZE, (self.player_view_distance * 2 + 3) * CELL_SIZE, obs_channels),
            dtype=np.uint8
        )

    def _in_vision(self, player_id, x, y):
        """
        Returns if given co-ordinates are within vision of the given player or not.
        """

        assert 0 <= player_id < self.n_players, "Invalid player_id"

        px, py = self.player_location[player_id]

        return max(abs(px - x), abs(py - y)) <= self.player_view_distance

    def player_at_pos(self, x, y, include_dead = False):
        """
        Returns first ID of player at given position or -1 if none
        :param x:
        :param y:
        :return:
        """

        # todo: this could be a lot faster, maybe we need a lookup?

        for player_id in range(self.n_players):
            if (include_dead or not self.is_dead(player_id)) and tuple(self.player_location[player_id]) == (x,y):
                return player_id
        return -1

    def _move_player(self, player_id, dx, dy):
        """
        Moves player given offset.
        Players may occupy cells with other players, but if they move outside of the map they will be clipped back in.
        """

        assert 0 <= player_id < self.n_players, "Invalid player_id"

        x,y = self.player_location[player_id]
        x = np.clip(x + dx, 0, self.map_width-1)
        y = np.clip(y + dy, 0, self.map_height-1)

        if self.player_at_pos(x, y) == -1:
            # can not move on-top of other players.
            self.player_location[player_id] = x, y

        # update general vision
        if self._in_vision(player_id, *self.general_location):

            # standing on general stops vision
            player_blocking = False
            for i in range(self.n_players):
                if tuple(self.player_location[i]) == (x, y):
                    player_blocking = True
            
            self.player_seen_general[player_id] = not player_blocking

    def is_dead(self, player_id):
        return self.player_health[player_id] <= 0

    def step(self, actions):
        """
        Perform game step
        :param actions: np array of actions of dims [n_players]
        :return: observations, rewards, dones, infos
        """
        assert len(actions) == self.n_players, "Invalid number of players"

        rescue_counter = 0
        green_tree_harvest_counter = 0

        rewards = np.zeros([self.n_players], dtype=np.float)
        dones = np.zeros([self.n_players], dtype=np.bool)

        # dead men can't act
        for player_id in range(self.n_players):
            if self.player_health[player_id] <= 0:
                actions[player_id] = self.ACTION_NOOP
            else:
                self.stats_actions[self.player_team[player_id]] += 1

        # apply actions, we process actions in the following order..
        # shooting
        # moving + acting

        red_team_good_kills = 0
        blue_team_good_kills = 0

        # shooting
        for player_id in range(self.n_players):

            if actions[player_id] in self.SHOOT_ACTIONS:

                self.stats_shots_fired[self.player_team[player_id]] += 1

                index = actions[player_id] - self.ACTION_SHOOT_UP
                x, y = self.player_location[player_id]

                for j in range(self.player_shoot_range):
                    # check location

                    x += self.DX[index]
                    y += self.DY[index]

                    # check other players
                    other_player_id = self.player_at_pos(x, y)
                    if other_player_id >= 0:
                        # a soldier was hit...
                        self._damage_player(other_player_id, np.random.randint(1,6) + np.random.randint(1,6))
                        if self.player_health[other_player_id] <= 0:
                            self.stats_deaths[self.player_team[other_player_id]] += 1
                            self.stats_kills[self.player_team[player_id]] += 1
                            # we killed the target player
                            if self.player_team[player_id] == self.TEAM_RED and self.player_team[other_player_id] == self.TEAM_BLUE:
                                red_team_good_kills += 1
                            elif self.player_team[player_id] == self.TEAM_BLUE and self.player_team[other_player_id] == self.TEAM_RED:
                                blue_team_good_kills += 1

                        self.stats_player_hit[self.player_team[player_id], self.player_team[other_player_id]] += 1
                        break

                    # check general
                    if (x, y) == self.general_location:
                        # general was hit
                        self.general_health -= (np.random.randint(1, 6) + np.random.randint(1, 6))
                        self.stats_general_shot[self.player_team[player_id]] += 1
                        self._needs_repaint = True
                        break

        # repeat dead men can't act as players might have been killed above
        for player_id in range(self.n_players):
            if self.player_health[player_id] <= 0:
                actions[player_id] = self.ACTION_NOOP

        # if we are within 1 range of general rescue him automatically
        for player_id in range(self.n_players):
            if self.player_team[[player_id]] == self.TEAM_BLUE and not self.is_dead(player_id):
                dx = self.player_location[player_id][0] - self.general_location[0]
                dy = self.player_location[player_id][1] - self.general_location[1]
                if abs(dx) + abs(dy) <= 1:
                    rescue_counter += 1

        # acting
        for player_id in range(self.n_players):
            if actions[player_id] == self.ACTION_ACT:
                self.stats_times_acted[self.player_team[player_id]] += 1
                px, py = self.player_location[player_id]
                if self.map[(px, py)] == self.MAP_TREE:
                    self.stats_tree_harvested[self.player_team[player_id]] += 1
                    if self.player_team[player_id] == self.TEAM_GREEN:
                        green_tree_harvest_counter += 1
                    self.map[(px, py)] = self.MAP_GRASS
                    self._needs_repaint = True

        # moving
        for player_id in range(self.n_players):
            if actions[player_id] in [self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN, self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT]:
                self.stats_times_moved[self.player_team[player_id]] += 1
                index = actions[player_id] - self.ACTION_MOVE_UP
                self._move_player(player_id, self.DX[index], self.DY[index])

        # generate points
        general_killed = self.general_health <= 0
        general_rescued = rescue_counter >= 2 and not general_killed
        game_timeout = self.counter >= self.timeout

        team_rewards = np.zeros([3], dtype=np.int)

        team_rewards[self.TEAM_GREEN] += green_tree_harvest_counter
        if self.easy_rewards:
            team_rewards[self.TEAM_RED] += red_team_good_kills
            team_rewards[self.TEAM_BLUE] += blue_team_good_kills
            team_rewards[self.TEAM_RED] -= blue_team_good_kills
            team_rewards[self.TEAM_BLUE] -= red_team_good_kills
        if general_killed:
            team_rewards[self.TEAM_RED] += 10
            team_rewards[self.TEAM_BLUE] -= 10
        elif general_rescued:
            team_rewards[self.TEAM_RED] -= 10
            team_rewards[self.TEAM_BLUE] += 10

        for player_id in range(self.n_players):
            rewards[player_id] = team_rewards[self.player_team[player_id]]

        self.team_scores += team_rewards

        all_players_dead = all(self.player_health[i] <= 0 for i in range(self.n_players))

        # send done notifications to players who are dead
        # for the moment just end everything once everyone's dead...
        # for i in range(self.n_players):
        #   dones[i] = self.player_health[i] <= 0

        # game ends for everyone under these outcomes
        if general_killed or general_rescued or game_timeout or all_players_dead:
            self.write_stats_to_log()
            dones[:] = True

        obs = self._get_observations()
        infos = [{} for _ in range(self.n_players)]

        self.counter += 1
        self.player_last_action = actions[:]

        rewards *= self.REWARD_SCALE

        return obs, rewards, dones, infos

    def _damage_player(self, player_id, damage):
        """
        Causes player to receive given damage.
        :param player_id:
        :param damage:
        :return:
        """
        self.player_health[player_id] = max(0, self.player_health[player_id] - damage)

    def _get_observations(self):
        return [self._get_player_observation(player_id) for player_id in range(self.n_players)]

    def _draw_tile(self, obs, x, y, c):
        dx, dy = x*CELL_SIZE, y*CELL_SIZE
        obs[dx:dx+CELL_SIZE, dy:dy+CELL_SIZE, :3] = c

    def _draw_soldier(self, obs: np.ndarray, player_id: int, team_colors=False, highlight=False, padding=(0, 0)):

        if self.player_health[player_id] <= 0:
            ring_color = self.COLOR_DEAD
        elif highlight:
            ring_color = self.COLOR_HIGHLIGHT
        else:
            ring_color = self.TEAM_COLOR[self.player_team[player_id]] if team_colors else self.COLOR_NEUTRAL

        fire_color = self.COLOR_FIRE
        inner_color = self.PLAYER_COLOR[player_id]

        x,y = self.player_location[player_id]

        dx, dy = (x+padding[0]) * CELL_SIZE + 1, (y+padding[1]) * CELL_SIZE + 1

        obs[dx - 1:dx + 2, dy - 1:dy + 2, :3] = ring_color
        obs[dx, dy, :3] = inner_color
        action = self.player_last_action[player_id]
        if action == self.ACTION_SHOOT_UP:
            obs[dx + 0, dy - 1, :3] = fire_color
        elif action == self.ACTION_SHOOT_LEFT:
            obs[dx - 1, dy + 0, :3] = fire_color
        elif action == self.ACTION_SHOOT_RIGHT:
            obs[dx + 1, dy + 0, :3] = fire_color
        elif action == self.ACTION_SHOOT_DOWN:
            obs[dx + 0, dy + 1, :3] = fire_color

    def _draw_general(self, obs):
        x, y = self.general_location
        dx, dy = x * CELL_SIZE, y * CELL_SIZE
        c = self.COLOR_GENERAL if self.general_health > 0 else self.COLOR_DEAD

        obs[dx + 1, dy + 1, :3] = c
        obs[dx + 2, dy + 1, :3] = c
        obs[dx + 0, dy + 1, :3] = c
        obs[dx + 1, dy + 2, :3] = c
        obs[dx + 1, dy + 0, :3] = c

    def _get_map(self):
        """
        Returns a map with optional positional encoding.
        Map is only redrawn as needed (e.g. when trees are removed).
        :return:
        """

        if not self._needs_repaint:
            return self._map_cache.copy()

        obs = np.zeros((self.map_width * CELL_SIZE, self.map_height * CELL_SIZE, 3), dtype=np.uint8)
        obs[:, :, :] = self.COLOR_GRASS

        # paint trees
        for x in range(self.map_width):
            for y in range(self.map_height):
                if self.map[x, y] == self.MAP_TREE:
                    self._draw_tile(obs, x, y, self.COLOR_TREE)

        # paint general
        self._draw_general(obs)

        # positional encoding
        position_obs = None
        if self.location_encoding == "none":
            pass
        elif self.location_encoding == "abs":
            x = np.linspace(0, 1, obs.shape[0])
            y = np.linspace(0, 1, obs.shape[1])
            xs, ys = np.meshgrid(x, y)
            position_obs = np.stack([xs, ys], axis=-1)
            position_obs = np.asarray(position_obs * 255, dtype=np.uint8)
        elif self.location_encoding == "sin":
            x = np.linspace(0, 2*math.pi, obs.shape[0])
            y = np.linspace(0, 2*math.pi, obs.shape[1])
            xs, ys = np.meshgrid(x, y)
            position_obs = []
            for layer in range(SIN_CHANNELS):
                freq = (2 ** (layer//2))
                f = np.sin if (layer // 2) % 2 == 0 else np.cos
                result = f((xs if layer % 2 == 0 else ys)*freq)
                result = np.asarray((result + 1) / 2 * 255, dtype=np.uint8)
                position_obs.append(result)
            position_obs = np.stack(position_obs, axis=-1)
        else:
            raise Exception(f"Invalid location encoding {self.location_encoding} use [none|sin|abs].")

        if position_obs is not None:
            obs = np.concatenate([obs, position_obs], axis=2)

        # padding (makes cropping easier...)
        self._map_padding = (self.player_view_distance + 1) # +1 for border
        padding = (self._map_padding*CELL_SIZE, self._map_padding*CELL_SIZE)
        obs = np.pad(obs, (padding, padding, (0, 0)), mode="constant")

        self._map_cache = obs
        self._needs_repaint = False
        return self._map_cache.copy()

    def write_log_buffer(self):
        """
        Write buffer to disk
        :return:
        """

        # append outcome to a log
        log_filename = f"{self.log_folder}/env.{self.id}.csv"

        if not os.path.exists(log_filename):
            with open(log_filename, "w") as f:
                f.write("game_counter, game_length, score_red, score_green, score_blue, " +
                        "stats_player_hit, stats_deaths, stats_kills, stats_general_shot, stats_tree_harvested, " +
                        "stats_shots_fired, stats_times_moved, stats_times_acted, stats_actions, player_count\n")

        with open(log_filename, "a+") as f:
            for output_string in LOG_BUFFER:
                f.write(output_string + "\n")

        LOG_BUFFER.clear()

    def write_stats_to_log(self):

        stats = [
            self.stats_player_hit,
            self.stats_deaths,
            self.stats_kills,
            self.stats_general_shot,
            self.stats_tree_harvested,
            self.stats_shots_fired,
            self.stats_times_moved,
            self.stats_times_acted,
            self.stats_actions
        ]

        def nice_print(x):
            return " ".join(str(i) for i in x.reshape(-1))

        output_string = ",".join(
            str(x) for x in [
                self.game_counter,
                self.counter,
                *self.team_scores,
                *(nice_print(x) for x in stats),
                self.n_players
            ]
        )

        LOG_BUFFER.append(output_string)

        if len(LOG_BUFFER) > 10:
            self.write_log_buffer()

    def _get_player_observation(self, player_id):
        """
        Generates a copy of the map with local observations given given player.
        :param player_id: player's perspective, -1 is global view
        :return:
        """

        obs = self._get_map()

        # paint soldiers, living over dead
        for i in range(self.n_players):

            player_visible = (player_id == -1) or self._in_vision(player_id, *self.player_location[i])

            if player_visible and self.is_dead(i):
                self._draw_soldier(
                    obs,
                    i,
                    team_colors=(not self.hidden_roles) or (i == player_id) or (player_id == -1),
                    padding=(self._map_padding, self._map_padding)
                )
        for i in range(self.n_players):

            player_visible = (player_id == -1) or self._in_vision(player_id, *self.player_location[i])

            if player_visible and not self.is_dead(i):
                self._draw_soldier(
                    obs,
                    i,
                    team_colors=(not self.hidden_roles) or (i == player_id) or (player_id == -1),
                    padding=(self._map_padding, self._map_padding)
                )

        # ego centric view
        if player_id >= 0:
            # get our local view
            left = (self._map_padding + self.player_location[player_id][0] - (self.player_view_distance + 1)) * CELL_SIZE
            right = (self._map_padding + self.player_location[player_id][0] + (self.player_view_distance + 2)) * CELL_SIZE
            top = (self._map_padding + self.player_location[player_id][1] - (self.player_view_distance + 1)) * CELL_SIZE
            bottom = (self._map_padding + self.player_location[player_id][1] + (self.player_view_distance + 2)) * CELL_SIZE
            obs = obs[left:right, top:bottom, :]
        else:
            # just remove padding
            padding = self._map_padding * CELL_SIZE
            obs = obs[padding:-padding, padding:-padding, :]

        if player_id >= 0:
            # blank out frame
            obs[:3, :, :3] = 64
            obs[-3:, :, :3] = 64
            obs[:, :3, :3] = 64
            obs[:, -3:, :3] = 64
            # mark top and bottom with time and health
            obs[3:-3, :3, :3] = int(self.counter / self.timeout * 255)
            obs[3:-3, -3:, 0:2] = int(self.player_health[player_id] / PLAYER_MAX_HEALTH * 255)

            # darken edges a little
            # this is really just for aesthetics
            obs[:1, :, :3] = 0
            obs[-1:, :, :3] = 0
            obs[:, :1, :3] = 0
            obs[:, -1:, :3] = 0

        # show general off-screen location
        if player_id >= 0 and self.player_seen_general[player_id]:

            dx = self.general_location[0] - self.player_location[player_id][0]
            dy = self.general_location[1] - self.player_location[player_id][1]

            if abs(dx) > self.player_view_distance or abs(dy) > self.player_view_distance:
                dx += self.player_view_distance
                dy += self.player_view_distance
                dx = np.clip(dx, -1, self.player_view_distance * 2 + 1)
                dy = np.clip(dy, -1, self.player_view_distance * 2 + 1)
                self._draw_tile(obs, dx+1, dy+1, self.COLOR_GENERAL) #not sure about these +1...

        if player_id >= 0:
            assert obs.shape == self.observation_space.shape, \
                f"Invalid observation crop, found {obs.shape} expected {self.observation_space.shape}."

        return obs

    def reset(self):
        """
        Reset game.
        :return: observations
        """

        # general location
        self.general_location = (np.random.randint(1, self.map_width - 2), np.random.randint(1, self.map_height - 2))
        self.general_health = self.initial_general_health

        self._needs_repaint = True

        # create map
        self.map = np.zeros((self.map_width, self.map_height), dtype=np.uint8) + 1

        all_locations = list(itertools.product(range(self.map_width), range(self.map_height)))
        idxs = np.random.choice(len(all_locations), size=self.n_trees, replace=False)
        for loc in [all_locations[idx] for idx in idxs]:
            self.map[loc] = 2

        # reset stats
        self.stats_player_hit *= 0
        self.stats_deaths *= 0
        self.stats_kills *= 0
        self.stats_general_shot *= 0
        self.stats_tree_harvested *= 0
        self.stats_shots_fired *= 0
        self.stats_times_moved *= 0
        self.stats_times_acted *= 0
        self.stats_actions *= 0

        # initialize players to random start locations, and assign initial health
        general_filter = lambda p: abs(p[0] - self.general_location[0]) > 3 and abs(p[1] - self.general_location[1]) > 3

        valid_start_locations = list(filter(general_filter, all_locations))
        start_locations = [valid_start_locations[idx] for idx in np.random.choice(len(valid_start_locations), size=self.n_players, replace=False)]

        self.player_location *= 0
        self.player_health *= 0
        self.player_seen_general *= 0
        self.player_team *= 0

        self.counter = 0
        self.game_counter += 1

        self.team_scores *= 0

        teams = [self.TEAM_RED] * self.n_players_red + [self.TEAM_GREEN] * self.n_players_green + [self.TEAM_BLUE] * self.n_players_blue
        np.random.shuffle(teams)
        self.player_team[:] = teams

        for i in range(self.n_players):
            self.player_location[i] = start_locations[i]
            self.player_health[i] = PLAYER_MAX_HEALTH
            self.player_seen_general[i] = self.player_team[i] == self.TEAM_BLUE or self.general_always_visible
            # this will update seen_general if player is close
            # which normally doesn't happen as I make sure players do not start close to the general
            self._move_player(i, 0, 0)

        return self._get_observations()

    @property
    def n_players(self):
        return self.n_players_red + self.n_players_green + self.n_players_blue

    def _render_human(self):
        raise NotImplemented("Sorry tile-map rendering not implemented yet")

    def _process_obs(self, obs, use_location=False):
        """
        Converts observation into RGBobs[:, :, 1]
        :return:
        """

        obs = obs.copy()

        if use_location:
            # get location embedding channels (if any exist)
            _, _, channels = obs.shape
            obs[:, :, :3] *= 0
            # stub average location channels
            if channels <= 6:
                # take last 3
                obs = obs[:, :, -3:]
            else:
                # average all location channels
                obs[:, :, 0] = np.mean(obs[:, :, 3:], axis=2)
                obs[:, :, 1] = obs[:, :, 0]

            obs = obs[:, :, :3]
        else:
            # standard RGB
            obs = obs[:, :, :3]

        return obs

    def _draw(self, frame, x, y, image):
        w,h,_ = image.shape
        frame[x:x+w, y:y+h] = image

    def _render_rgb(self, use_location=False):
        """
        Render out a frame
        :param use_location:
        :return:
        """

        # todo, don't assume 4 players, and have option for including player observations

        global_frame = self._process_obs(self._get_player_observation(-1), use_location)

        player_frames = [
            self._process_obs(self._get_player_observation(player_id), use_location) for player_id in range(self.n_players)
        ]

        gw, gh, _ = global_frame.shape
        pw, ph, _ = player_frames[0].shape

        grid_width = 3
        grid_height = 2

        frame = np.zeros((gw+pw*grid_width, max(gh, ph*grid_height), 3), np.uint8)
        self._draw(frame, 0, 0, global_frame)
        i = 0
        for x in range(grid_width):
            for y in range(grid_height):
                # draw a darker version of player observations so they don't distract too much
                self._draw(frame, gw + pw*x, ph*y, player_frames[i] * 0.75)
                i = i + 1

        # add video padding
        padding = (CELL_SIZE, CELL_SIZE)
        frame = np.pad(frame, (padding, padding, (0, 0)), mode="constant")

        frame = frame.swapaxes(0, 1) # I'm using x,y, but video needs y,x

        return frame

    def render(self, mode='human', use_location=False):
        """ render environment """
        if mode == 'human':
            return self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb(use_location=use_location)
        else:
            raise ValueError(f"Invalid render mode {mode}")


# all envs share this log
LOG_BUFFER = []