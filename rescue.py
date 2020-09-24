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
import time

from stable_baselines.common.misc_util import mpi_rank_or_zero

from MARL import MultiAgentEnv

CELL_SIZE = 3
SIN_CHANNELS = 10 # 10 channels gets a frequencies of 2^5, which suits maps around 32 tiles in width/height

class RescueTheGeneralScenario():
    def __init__(self, **kwargs):

        # defaults
        self.n_trees = 10
        self.map_width = 32
        self.map_height = 32
        self.player_view_distance = 5
        self.player_shoot_range = 4
        self.timeout = 1000
        self.general_always_visible = False
        self.general_initial_health = 10
        self.player_initial_health = 10
        self.location_encoding = "abs"  # none | sin | abs
        self.player_counts = (2, 2, 2)
        self.hidden_roles = True
        self.easy_rewards = True  # enables some easy rewards, such as killing enemy players.

        self.description = ""

        # overrides
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        result = []
        for k,v in vars(self).items():
            result.append(f"{k:<24} = {v}")
        return "\n".join(result)


class RTG_Player():

    def __init__(self, id, scenario):
        self.id = id
        self.x, self.y = int(), int()
        self.health = int()
        self.team = int()
        self.action = int()
        self.scenario = scenario

    @property
    def team_color(self):
        return RescueTheGeneralEnv.TEAM_COLOR[self.team]

    @property
    def id_color(self):
        return RescueTheGeneralEnv.ID_COLOR[self.id]

    @property
    def is_dead(self):
        return self.health <= 0

    def in_vision(self, x, y):
        """
        Returns if given co-ordinates are within vision of the given player or not.
        """
        return max(abs(self.x - x), abs(self.y - y)) <= self.scenario.player_view_distance

    def damage(self, damage):
        """
        Causes player to receive given damage.
        :param player_id:
        :param damage:
        :return:
        """
        self.health = max(0, self.health - damage)


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

    ID_COLOR = np.asarray(
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

    SCENARIOS = {
        "default": {
            "description": "This is the default scenario."
        },

        "red2": {
            "description": "Two red players must find and kill general on small map.",
            "map_width": 24,
            "map_height": 24,
            "player_counts": (2, 0, 0),
            "hidden_roles": False,
        }
    }

    def __init__(self, scenario="default"):
        super().__init__()

        self.env_create_time = time.time()

        self.log_folder = "./"
        self._needs_repaint = True

        # setup our scenario
        scenario_kwargs = self.SCENARIOS[scenario]
        self.scenario = RescueTheGeneralScenario(**scenario_kwargs)

        self.id = mpi_rank_or_zero()
        self.counter = 0
        self.game_counter = 0

        self._log_buffer = []

        self.general_location = (0,0)
        self.general_health = 0

        self.players = [RTG_Player(id, self.scenario) for id in range(self.n_players)]

        self.team_scores = np.zeros([3], dtype=np.int)

        self.reward_logging = True # logs rewards to txt file

        # create map, and a lookup (just for optimization
        self.map = np.zeros((self.scenario.map_width, self.scenario.map_height), dtype=np.int)
        self.player_lookup = np.zeros((self.scenario.map_width, self.scenario.map_height), dtype=np.int)

        self.stats_player_hit = np.zeros((3,3), dtype=np.int) # which teams killed who
        self.stats_deaths = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_kills = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_general_shot = np.zeros((3,), dtype=np.int)  # which teams shot general
        self.stats_tree_harvested = np.zeros((3,), dtype=np.int)  # which teams harvested trees
        self.stats_shots_fired = np.zeros((3,), dtype=np.int)  # how many times each team shot
        self.stats_times_moved = np.zeros((3,), dtype=np.int)  # how many times each team moved
        self.stats_times_acted = np.zeros((3,), dtype=np.int)  # how many times each team acted
        self.stats_actions = np.zeros((3,), dtype=np.int)  # how actions this team could have performed (sum_t(agents_alive))

        self.action_space = gym.spaces.Discrete(10)

        obs_channels = 3
        if self.scenario.location_encoding == "none":
            pass
        elif self.scenario.location_encoding == "sin":
            obs_channels += SIN_CHANNELS
        elif self.scenario.location_encoding == "abs":
            obs_channels += 2
        else:
            raise Exception(f"Invalid location encoding {self.scenario.location_encoding} use [none|sin|abs].")

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=((self.scenario.player_view_distance * 2 + 3) * CELL_SIZE, (self.scenario.player_view_distance * 2 + 3) * CELL_SIZE, obs_channels),
            dtype=np.uint8
        )

    def player_at_pos(self, x, y, include_dead = False):
        """
        Returns first player at given position or None if no players are on that tile
        :param x:
        :param y:
        :return:
        """

        if include_dead:
            for player in self.players:
                if player.x == x and player.y == y:
                    return player
        else:
            # lookup map only includes living players
            id = self.player_lookup[x, y]
            return self.players[id] if id >= 0 else None

    def move_player(self, player, dx, dy):
        """
        Moves player given offset.
        Players may occupy cells with other players, but if they move outside of the map they will be clipped back in.
        """

        new_x = min(max(player.x + dx, 0), self.scenario.map_width - 1)
        new_y = min(max(player.y + dy, 0), self.scenario.map_height - 1)

        if self.player_at_pos(player.x, player.y) is None:
            # can not move on-top of other players.
            self.player_lookup[player.x, player.y] = -1
            self.player_lookup[new_x, new_y] = player.id
            player.x = new_x
            player.y = new_y

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

        # assign actions to players
        for id, player in enumerate(self.players):
            player.action = actions[id]

        # apply actions, we process actions in the following order..
        # shooting
        # moving + acting

        red_team_good_kills = 0
        blue_team_good_kills = 0

        # shooting
        for player in self.players:

            if player.is_dead:
                continue

            if player.action in self.SHOOT_ACTIONS:

                self.stats_shots_fired[player.team] += 1

                index = player.action - self.ACTION_SHOOT_UP
                x = player.x
                y = player.y

                for j in range(self.scenario.player_shoot_range):
                    # check location

                    x += self.DX[index]
                    y += self.DY[index]

                    if x < 0 or x >= self.scenario.map_width or y < 0 or y >= self.scenario.map_height:
                        break

                    # check other players
                    other_player = self.player_at_pos(x, y)
                    if other_player is not None:
                        # a soldier was hit...
                        other_player.damage(np.random.randint(1,6) + np.random.randint(1,6))
                        if other_player.is_dead:
                            # we killed the target player
                            self.stats_deaths[other_player.team] += 1
                            self.stats_kills[player.team] += 1
                            if player.team == self.TEAM_RED and other_player.team == self.TEAM_BLUE:
                                red_team_good_kills += 1
                            elif player.team == self.TEAM_BLUE and other_player.team == self.TEAM_RED:
                                blue_team_good_kills += 1
                            self.player_lookup[player.x, player.y] = -1 # remove player from lookup

                        self.stats_player_hit[player.team, other_player.team] += 1
                        break

                    # check general
                    if (x, y) == self.general_location:
                        # general was hit
                        self.general_health -= (np.random.randint(1, 6) + np.random.randint(1, 6))
                        self.stats_general_shot[player.team] += 1
                        self._needs_repaint = True
                        break

        # if we are within 1 range of general rescue him automatically
        for player in self.players:

            if player.is_dead:
                continue

            if player.team == self.TEAM_BLUE and not player.is_dead:
                dx = player.x - self.general_location[0]
                dy = player.y - self.general_location[1]
                if abs(dx) + abs(dy) <= 1:
                    rescue_counter += 1

        # acting
        for player in self.players:

            if player.is_dead:
                continue

            if player.action == self.ACTION_ACT:
                self.stats_times_acted[player.team] += 1
                if self.map[(player.x, player.y)] == self.MAP_TREE:
                    self.stats_tree_harvested[player.team] += 1
                    if player.team == self.TEAM_GREEN:
                        green_tree_harvest_counter += 1
                    self.map[(player.x, player.y)] = self.MAP_GRASS
                    self._needs_repaint = True

        # moving
        for player in self.players:

            if player.is_dead:
                continue

            if player.action in [self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN, self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT]:
                self.stats_times_moved[player.team] += 1
                index = player.action - self.ACTION_MOVE_UP
                self.move_player(player, self.DX[index], self.DY[index])

        # generate points
        general_killed = self.general_health <= 0
        general_rescued = rescue_counter >= 2 and not general_killed
        game_timeout = self.counter >= self.scenario.timeout

        team_rewards = np.zeros([3], dtype=np.int)

        team_rewards[self.TEAM_GREEN] += green_tree_harvest_counter
        if self.scenario.easy_rewards:
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

        for id, player in enumerate(self.players):
            rewards[id] = team_rewards[player.team]

        self.team_scores += team_rewards

        all_players_dead = all(player.is_dead for player in self.players)

        # count actions
        for player in self.players:
            if not player.is_dead:
                self.stats_actions[player.team] += 1

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

        rewards *= self.REWARD_SCALE

        return obs, rewards, dones, infos

    def _get_observations(self):
        return [self._get_player_observation(player_id) for player_id in range(self.n_players)]

    def draw_tile(self, obs, x, y, c):
        dx, dy = x*CELL_SIZE, y*CELL_SIZE
        obs[dx:dx+CELL_SIZE, dy:dy+CELL_SIZE, :3] = c

    def _draw_soldier(self, obs: np.ndarray, player: RTG_Player, team_colors=False, highlight=False, padding=(0, 0)):

        if player.is_dead:
            ring_color = self.COLOR_DEAD
        elif highlight:
            ring_color = self.COLOR_HIGHLIGHT
        else:
            ring_color = self.TEAM_COLOR[player.team] if team_colors else self.COLOR_NEUTRAL

        fire_color = self.COLOR_FIRE
        inner_color = player.id_color

        dx, dy = (player.x+padding[0]) * CELL_SIZE + 1, (player.y+padding[1]) * CELL_SIZE + 1

        obs[dx - 1:dx + 2, dy - 1:dy + 2, :3] = ring_color
        obs[dx, dy, :3] = inner_color

        if player.action == self.ACTION_SHOOT_UP:
            obs[dx + 0, dy - 1, :3] = fire_color
        elif player.action == self.ACTION_SHOOT_LEFT:
            obs[dx - 1, dy + 0, :3] = fire_color
        elif player.action == self.ACTION_SHOOT_RIGHT:
            obs[dx + 1, dy + 0, :3] = fire_color
        elif player.action == self.ACTION_SHOOT_DOWN:
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

        obs = np.zeros((self.scenario.map_width * CELL_SIZE, self.scenario.map_height * CELL_SIZE, 3), dtype=np.uint8)
        obs[:, :, :] = self.COLOR_GRASS

        # paint trees
        for x in range(self.scenario.map_width):
            for y in range(self.scenario.map_height):
                if self.map[x, y] == self.MAP_TREE:
                    self.draw_tile(obs, x, y, self.COLOR_TREE)

        # paint general
        self._draw_general(obs)

        # positional encoding
        position_obs = None
        if self.scenario.location_encoding == "none":
            pass
        elif self.scenario.location_encoding == "abs":
            x = np.linspace(0, 1, obs.shape[0])
            y = np.linspace(0, 1, obs.shape[1])
            xs, ys = np.meshgrid(x, y)
            position_obs = np.stack([xs, ys], axis=-1)
            position_obs = np.asarray(position_obs * 255, dtype=np.uint8)
        elif self.scenario.location_encoding == "sin":
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
            raise Exception(f"Invalid location encoding {self.scenario.location_encoding} use [none|sin|abs].")

        if position_obs is not None:
            obs = np.concatenate([obs, position_obs], axis=2)

        # padding (makes cropping easier...)
        self._map_padding = (self.scenario.player_view_distance + 1) # +1 for border
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
                        "stats_shots_fired, stats_times_moved, stats_times_acted, stats_actions, player_count," +
                        "wall_time, date_time" +
                        "\n")

        with open(log_filename, "a+") as f:
            for output_string in LOG_BUFFER:
                f.write(output_string + "\n")

        LOG_BUFFER.clear()

        global LOG_BUFFER_LAST_WRITE_TIME
        LOG_BUFFER_LAST_WRITE_TIME = time.time()

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

        time_since_env_started = time.time() - self.env_create_time

        output_string = ",".join(
            str(x) for x in [
                self.game_counter,
                self.counter,
                *self.team_scores,
                *(nice_print(x) for x in stats),
                self.n_players,
                time_since_env_started,
                time.time()

            ]
        )

        LOG_BUFFER.append(output_string)

        # write to buffer every 120 seconds
        time_since_last_log_write = time.time() - LOG_BUFFER_LAST_WRITE_TIME
        if time_since_last_log_write > 120:
            self.write_log_buffer()

    def _get_player_observation(self, observer_id):
        """
        Generates a copy of the map with local observations given given player.
        :param observer_id: player's perspective, -1 is global view
        :return:
        """

        obs = self._get_map()

        observer = self.players[observer_id] if (observer_id != -1) else None

        # paint soldiers, living over dead
        for player in self.players:
            player_is_visible = (observer is None) or observer.in_vision(player.x, player.y)
            if player.is_dead and player_is_visible:
                self._draw_soldier(
                    obs,
                    player,
                    team_colors=(not self.scenario.hidden_roles) or (player == observer) or (observer is None),
                    padding=(self._map_padding, self._map_padding)
                )
        for player in self.players:
            player_is_visible = (observer is None) or observer.in_vision(player.x, player.y)
            if not player.is_dead and player_is_visible:
                self._draw_soldier(
                    obs,
                    player,
                    team_colors=(not self.scenario.hidden_roles) or (player == observer) or (observer is None),
                    padding=(self._map_padding, self._map_padding)
                )

        # ego centric view
        if observer_id >= 0:
            # get our local view
            left = (self._map_padding + observer.x - (self.scenario.player_view_distance + 1)) * CELL_SIZE
            right = (self._map_padding + observer.x + (self.scenario.player_view_distance + 2)) * CELL_SIZE
            top = (self._map_padding + observer.y - (self.scenario.player_view_distance + 1)) * CELL_SIZE
            bottom = (self._map_padding + observer.y + (self.scenario.player_view_distance + 2)) * CELL_SIZE
            obs = obs[left:right, top:bottom, :]
        else:
            # just remove padding
            padding = self._map_padding * CELL_SIZE
            obs = obs[padding:-padding, padding:-padding, :]

        if observer_id >= 0:
            # blank out frame
            obs[:3, :, :3] = 64
            obs[-3:, :, :3] = 64
            obs[:, :3, :3] = 64
            obs[:, -3:, :3] = 64
            # mark top and bottom with time and health
            obs[3:-3, :3, :3] = int(self.counter / self.scenario.timeout * 255)
            obs[3:-3, -3:, 0:2] = int(observer.health / self.scenario.player_initial_health * 255)

            # darken edges a little
            # this is really just for aesthetics
            obs[:1, :, :3] = 0
            obs[-1:, :, :3] = 0
            obs[:, :1, :3] = 0
            obs[:, -1:, :3] = 0

        # show general off-screen location
        if (observer is not None) and (observer.team == self.TEAM_BLUE or self.scenario.general_always_visible):

            dx = self.general_location[0] - observer.x
            dy = self.general_location[1] - observer.y

            if abs(dx) > self.scenario.player_view_distance or abs(dy) > self.scenario.player_view_distance:
                dx += self.scenario.player_view_distance
                dy += self.scenario.player_view_distance
                dx = max(min(dx, -1), self.scenario.player_view_distance * 2 + 1)
                dy = max(min(dy, -1), self.scenario.player_view_distance * 2 + 1)
                self.draw_tile(obs, dx + 1, dy + 1, self.COLOR_GENERAL)

        if observer_id >= 0:
            assert obs.shape == self.observation_space.shape, \
                f"Invalid observation crop, found {obs.shape} expected {self.observation_space.shape}."

        return obs

    def reset(self):
        """
        Reset game.
        :return: observations
        """

        # general location
        self.general_location = (np.random.randint(1, self.scenario.map_width - 2), np.random.randint(1, self.scenario.map_height - 2))
        self.general_health = self.scenario.general_initial_health

        self._needs_repaint = True

        # create map
        self.map[:, :] = 1
        self.player_lookup[:, :] = -1

        all_locations = list(itertools.product(range(self.scenario.map_width), range(self.scenario.map_height)))
        idxs = np.random.choice(len(all_locations), size=self.scenario.n_trees, replace=False)
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

        self.counter = 0
        self.game_counter += 1

        self.team_scores *= 0

        # assign teams
        teams = [self.TEAM_RED] * self.scenario.player_counts[0] + \
                [self.TEAM_GREEN] * self.scenario.player_counts[1] + \
                [self.TEAM_BLUE] * self.scenario.player_counts[2]

        np.random.shuffle(teams)
        for id, team in enumerate(teams):
            self.players[id].team = team

        for id, player in enumerate(self.players):
            player.x, player.y = start_locations[id]
            self.player_lookup[player.x, player.y] = id
            player.health = self.scenario.player_initial_health

        return self._get_observations()

    @property
    def n_players(self):
        return sum(self.scenario.player_counts)

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

        if 0 <= self.n_players <= 1:
            grid_width = 1
            grid_height = 1
        elif 2 <= self.n_players <= 4:
            grid_width = 2
            grid_height = 2
        elif 5 <= self.n_players <= 6:
            grid_width = 3
            grid_height = 2
        elif 7 <= self.n_players <= 8:
            grid_width = 4
            grid_height = 2
        elif self.n_players <= 16:
            grid_width = 4
            grid_height = 4
        else:
            grid_width = grid_height = math.ceil(self.n_players ** 0.5)

        frame = np.zeros((gw+pw*grid_width, max(gh, ph*grid_height), 3), np.uint8)
        self._draw(frame, 0, 0, global_frame)
        i = 0
        for x in range(grid_width):
            for y in range(grid_height):
                # draw a darker version of player observations so they don't distract too much
                if i < len(player_frames):
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
LOG_BUFFER_LAST_WRITE_TIME = time.time()