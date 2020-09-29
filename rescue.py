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

        self.n_trees = 20
        self.reward_per_tree = 0.5
        self.map_width = 48
        self.map_height = 48
        self.player_view_distance = 5
        self.player_shoot_range = 4
        self.timeout = 500
        self.general_always_visible = False
        self.general_initial_health = 10
        self.player_initial_health = 10
        self.location_encoding = "abs"  # none | sin | abs
        self.player_counts = (4, 4, 4)
        self.hidden_roles = True
        self.battle_royale = False   # removes general from game, and adds kill rewards
        self.enable_signals = True

        self.shooting_timeout = 3    # number of turns between shooting
        self.reveal_team_on_death = False

        self.description = "The full game"

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
        self.shooting_timeout = int()
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

    FIRE_COLOR = np.asarray([255, 255, 0], dtype=np.uint8)
    SIGNAL_COLOR = np.asarray([20, 20, 20], dtype=np.uint8)
    HIGHLIGHT_COLOR = np.asarray([180, 180, 50], dtype=np.uint8)
    GENERAL_COLOR = np.asarray([255, 255, 255], dtype=np.uint8)
    NEUTRAL_COLOR = np.asarray([64, 64, 64], dtype=np.uint8)
    GRASS_COLOR = np.asarray([0, 128, 0], dtype=np.uint8)
    TREE_COLOR = np.asarray([125, 255, 150], dtype=np.uint8)
    DEAD_COLOR = np.asarray([0, 0, 0], dtype=np.uint8)

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
    ACTION_SIGNAL_UP = 10
    ACTION_SIGNAL_DOWN = 11
    ACTION_SIGNAL_LEFT = 12
    ACTION_SIGNAL_RIGHT = 13

    SHOOT_ACTIONS = [ACTION_SHOOT_UP, ACTION_SHOOT_DOWN, ACTION_SHOOT_LEFT, ACTION_SHOOT_RIGHT]
    SIGNAL_ACTIONS = [ACTION_SIGNAL_UP, ACTION_SIGNAL_DOWN, ACTION_SIGNAL_LEFT, ACTION_SIGNAL_RIGHT]
    MOVE_ACTIONS = [ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT]

    DX = [0, 0, -1, +1]
    DY = [-1, 1, 0, 0]

    SCENARIOS = {
        "full": {
            "description": "This is the default scenario."
        },

        "8blue": {
            "description": "Eight blue two red, no green, default map",
            "player_counts": (8, 0, 2),
        },

        "red2": {
            "description": "Two red players must find and kill general on small map.",
            "map_width": 24,
            "map_height": 24,
            "player_counts": (2, 0, 0),
            "n_trees": 10,
            "reward_per_tree": 1,
            "hidden_roles": False,
            "timeout": 1000,
        },

        "green2": {
            "description": "Two green players must harvest trees uncontested on a small map.",
            "map_width": 24,
            "map_height": 24,
            "player_counts": (0, 2, 0),
            "n_trees": 10,
            "reward_per_tree": 1,
            "hidden_roles": False,
            "timeout": 1000,
        },

        "blue2": {
            "description": "Two blue players must rescue the general on a small map.",
            "map_width": 24,
            "map_height": 24,
            "player_counts": (0, 0, 2),
            "n_trees": 10,
            "reward_per_tree": 1,
            "hidden_roles": False,
            "timeout": 1000,
        },

        # the idea here is to try and learn the other players identity
        "royale": {
            "description": "Red vs Blue, two soldiers each, in hidden roles battle royale.",
            "map_width": 24,
            "map_height": 24,
            "player_counts": (2, 0, 2),
            "n_trees": 0,
            "hidden_roles": True,
            "battle_royale": True,
            "reveal_team_on_death": True
        }
    }

    def __init__(self, scenario="full"):
        super().__init__()

        self.action_space = gym.spaces.Discrete(14)

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
        self.general_health = int()
        self.general_closest_tiles_from_edge = int()
        self.blue_rewards_for_winning = int()

        self.players = [RTG_Player(id, self.scenario) for id in range(self.n_players)]

        self.team_scores = np.zeros([3], dtype=np.float)

        # create map, and a lookup (just for optimization
        self.map = np.zeros((self.scenario.map_width, self.scenario.map_height), dtype=np.int)
        self.player_lookup = np.zeros((self.scenario.map_width, self.scenario.map_height), dtype=np.int)

        self.stats_player_hit = np.zeros((3,3), dtype=np.int) # which teams killed who
        self.stats_deaths = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_kills = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_general_shot = np.zeros((3,), dtype=np.int)  # which teams shot general
        self.stats_tree_harvested = np.zeros((3,), dtype=np.int)  # which teams harvested trees

        self.stats_actions = np.zeros((3, self.action_space.n), dtype=np.int)
        self.stats_outcome = "" # outcome of game

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

            player = self.players[id] if id >= 0 else None

            if player is not None:
                assert player.x == x and player.y == y, "player_lookup has invalid value."

            return player

    def move_player(self, player, dx, dy):
        """
        Moves player given offset.
        Players may occupy cells with other players, but if they move outside of the map they will be clipped back in.
        """

        new_x = min(max(player.x + dx, 0), self.scenario.map_width - 1)
        new_y = min(max(player.y + dy, 0), self.scenario.map_height - 1)

        if self.player_at_pos(new_x, new_y) is None:
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

        green_tree_harvest_counter = 0

        rewards = np.zeros([self.n_players], dtype=np.float)
        dones = np.zeros([self.n_players], dtype=np.bool)
        infos = [{} for _ in range(self.n_players)]

        # assign actions to players / remove invalid actions
        for player in self.players:
            player.action = self.ACTION_NOOP if player.is_dead else actions[player.id]
            if player.shooting_timeout != 0 and player.action in self.SHOOT_ACTIONS:
                player.action = self.ACTION_NOOP

        # count actions
        for player in self.players:
            self.stats_actions[player.team, player.action] += 1

        # apply actions, we process actions in the following order..
        # shooting
        # moving + acting

        red_team_good_kills = 0
        blue_team_good_kills = 0
        team_self_kills = [0, 0, 0]
        team_deaths = [0, 0, 0]

        # -----------------------------------------
        # shooting
        # note living players will give us all players living before the shooting starts, so players killed during
        # combat still get to shoot this round
        for player in self.living_players:

            if player.action not in self.SHOOT_ACTIONS:
                continue

            index = player.action - self.ACTION_SHOOT_UP
            x = player.x
            y = player.y

            player.shooting_timeout = self.scenario.shooting_timeout

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
                        team_deaths[other_player.team] += 1

                        if player.team == other_player.team:
                            team_self_kills[player.team] += 1

                        if player.team == self.TEAM_RED and other_player.team == self.TEAM_BLUE:
                            red_team_good_kills += 1
                        elif player.team == self.TEAM_BLUE and other_player.team == self.TEAM_RED:
                            blue_team_good_kills += 1

                        self.player_lookup[other_player.x, other_player.y] = -1 # remove player from lookup

                    self.stats_player_hit[player.team, other_player.team] += 1
                    break

                # check general
                if not self.scenario.battle_royale and ((x, y) == self.general_location):
                    # general was hit
                    self.general_health -= (np.random.randint(1, 6) + np.random.randint(1, 6))
                    self.stats_general_shot[player.team] += 1
                    self._needs_repaint = True
                    break

        # reduce shooting time_out
        for player in self.players:
            player.shooting_timeout = max(0, player.shooting_timeout - 1)

        # -------------------------
        # action button
        general_has_been_moved = False
        general_is_closer_to_edge = False
        for player in self.living_players:

            if player.action != self.ACTION_ACT:
                continue

            # harvest a tree if we are standing on it
            if self.map[(player.x, player.y)] == self.MAP_TREE:
                self.stats_tree_harvested[player.team] += 1
                if player.team == self.TEAM_GREEN:
                    green_tree_harvest_counter += 1
                self.map[(player.x, player.y)] = self.MAP_GRASS
                self._needs_repaint = True
                continue

            # move general by one tile if we are standing next to them
            if not general_has_been_moved and abs(player.x - self.general_location[0]) + abs(player.y - self.general_location[1]) == 1:
                self.general_location = (player.x, player.y)
                # moving the general is a once per turn thing
                general_has_been_moved = True
                # award some score if general is closer to the edge than they used to be
                if self.general_tiles_from_edge < self.general_closest_tiles_from_edge:
                    self.general_closest_tiles_from_edge = self.general_tiles_from_edge
                    general_is_closer_to_edge = True

        # ------------------------
        # moving
        for player in self.living_players:

            if player.action in self.MOVE_ACTIONS:
                index = player.action - self.ACTION_MOVE_UP
                self.move_player(player, self.DX[index], self.DY[index])

        # ------------------------
        # generate points
        result_general_killed = self.general_health <= 0
        result_general_rescued = self.general_tiles_from_edge == 0
        result_game_timeout = self.counter >= self.scenario.timeout
        result_all_players_dead = all(player.is_dead for player in self.players)
        result_red_victory = False
        result_blue_victory = False
        result_green_victory = False

        team_rewards = np.zeros([3], dtype=np.float)
        team_players_alive = np.zeros([3], dtype=np.int)

        team_rewards[self.TEAM_GREEN] += green_tree_harvest_counter * self.scenario.reward_per_tree

        for player in self.players:
            if not player.is_dead:
                team_players_alive[player.team] += 1

        if np.sum(team_players_alive) == team_players_alive[self.TEAM_GREEN] and team_players_alive[self.TEAM_GREEN] >= 1:
            # check for harvesting complete
            unique, counts = np.unique(self.map, return_counts=True)
            if self.MAP_TREE not in unique:
                result_green_victory = True

        if general_is_closer_to_edge:
            # give a very small reward for moving general closer to the edge
            small_reward = (1/self.scenario.map_width)
            team_rewards[self.TEAM_BLUE] += small_reward
            self.blue_rewards_for_winning -= small_reward # make sure blue always gets the same number of points for winning

        if self.scenario.battle_royale:

            # battle royal has very different rewards

            # gain points for killing enemy players
            team_rewards[self.TEAM_RED] += red_team_good_kills
            team_rewards[self.TEAM_BLUE] += blue_team_good_kills

            # loose points for  deaths
            for team in range(3):
                team_rewards[team] -= team_deaths[team]

            # loose a lot of points if non-one wins
            if result_game_timeout:
                team_rewards[self.TEAM_RED] -= 5
                team_rewards[self.TEAM_GREEN] += 5
                team_rewards[self.TEAM_BLUE] -= 5

            # gain some points if all enemy players are eliminated
            red_players = sum(not player.is_dead for player in self.players if player.team == self.TEAM_RED)
            green_players = sum(not player.is_dead for player in self.players if player.team == self.TEAM_GREEN)
            blue_players = sum(not player.is_dead for player in self.players if player.team == self.TEAM_BLUE)

            if red_players > 0 and blue_players == 0:
                team_rewards[self.TEAM_RED] += 10
                result_red_victory = True
            if blue_players > 0 and red_players == 0:
                team_rewards[self.TEAM_BLUE] += 10
                result_blue_victory = True

        if result_general_killed:
            team_rewards[self.TEAM_RED] += 10
            team_rewards[self.TEAM_BLUE] -= 10
        elif result_general_rescued:
            team_rewards[self.TEAM_RED] -= 10
            team_rewards[self.TEAM_BLUE] += self.blue_rewards_for_winning

        for player in self.players:
            rewards[player.id] = team_rewards[player.team]

        self.team_scores += team_rewards

        # send done notifications to players who are dead
        for player in self.players:
            dones[player.id] = player.is_dead

        game_finished = result_general_killed or \
                        result_general_rescued or \
                        result_game_timeout or \
                        result_all_players_dead or \
                        result_red_victory or \
                        result_blue_victory or \
                        result_green_victory

        # game ends for everyone under these outcomes
        if game_finished:

            if result_general_killed:
                self.stats_outcome = "general_killed"
            elif result_general_rescued:
                self.stats_outcome = "general_rescued"
            elif result_game_timeout:
                self.stats_outcome = "timeout"
            elif result_all_players_dead:
                self.stats_outcome = "all_players_dead"
            elif result_red_victory:
                self.stats_outcome = "red_win" # royale wins
            elif result_blue_victory:
                self.stats_outcome = "blue_win"
            elif result_green_victory:
                self.stats_outcome = "green_win"

            self.write_stats_to_log()
            dones[:] = True

            for info in infos:
                # record the outcome in infos as it will be lost if environment is auto reset.
                info["outcome"] = self.stats_outcome

        obs = self._get_observations()

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
            if not self.scenario.hidden_roles or self.scenario.reveal_team_on_death:
                ring_color = (self.TEAM_COLOR[player.team] // 2 + self.DEAD_COLOR // 2)
            else:
                ring_color = self.DEAD_COLOR
        elif highlight:
            ring_color = self.HIGHLIGHT_COLOR
        else:
            ring_color = self.TEAM_COLOR[player.team] if team_colors else self.NEUTRAL_COLOR

        inner_color = player.id_color

        draw_x, draw_y = (player.x+padding[0]) * CELL_SIZE + 1, (player.y+padding[1]) * CELL_SIZE + 1

        obs[draw_x - 1:draw_x + 2, draw_y - 1:draw_y + 2, :3] = ring_color
        obs[draw_x, draw_y, :3] = inner_color

        if player.action in self.SHOOT_ACTIONS:
            index = player.action - self.ACTION_SHOOT_UP
            dx, dy = self.DX[index], self.DY[index]
            obs[draw_x + dx, draw_y + dy, :3] = self.FIRE_COLOR

        if self.scenario.enable_signals and player.action in self.SIGNAL_ACTIONS:
            index = player.action - self.ACTION_SIGNAL_UP
            dx, dy = self.DX[index], self.DY[index]
            obs[draw_x + dx, draw_y + dy, :3] = self.SIGNAL_COLOR

    def _draw_general(self, obs, padding=(0, 0)):

        if self.scenario.battle_royale:
            return

        x, y = self.general_location
        dx, dy = (x+padding[0]) * CELL_SIZE, (y+padding[1]) * CELL_SIZE
        c = self.GENERAL_COLOR if self.general_health > 0 else self.DEAD_COLOR

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
        obs[:, :, :] = self.GRASS_COLOR

        # paint trees
        for x in range(self.scenario.map_width):
            for y in range(self.scenario.map_height):
                if self.map[x, y] == self.MAP_TREE:
                    self.draw_tile(obs, x, y, self.TREE_COLOR)

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
                        "stats_player_hit, stats_deaths, stats_kills, stats_general_shot, stats_tree_harvested, stats_actions, " +
                        "player_count, result, wall_time, date_time" +
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
            self.stats_actions,
            np.asarray(self.scenario.player_counts, dtype=np.int)
        ]

        def nice_print(x):
            return " ".join(str(i) for i in x.reshape(-1))

        time_since_env_started = time.time() - self.env_create_time

        output_string = ",".join(
            str(x) for x in [
                self.game_counter, self.counter, *self.team_scores,
                *(nice_print(x) for x in stats),
                self.stats_outcome, time_since_env_started, time.time()
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

        # in the global view we paint general on-top of soldiers so we always know where he is
        self._draw_general(obs, (self._map_padding, self._map_padding))

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
            obs[:3, :, :3] = 32
            obs[-3:, :, :3] = 32
            obs[:, :3, :3] = 32
            obs[:, -3:, :3] = 32

            # bars for time, health, and shooting timeout
            frame_width, frame_height, _ = obs[3:-3, 3:-3, :].shape
            time_bar = int((self.scenario.timeout - self.counter) / self.scenario.timeout * frame_width)
            health_bar = int(observer.health / self.scenario.player_initial_health * frame_width)
            shooting_bar = int(observer.shooting_timeout / self.scenario.shooting_timeout * frame_width)

            obs[3:3 + time_bar, -3, :3] = (255, 255, 128)
            obs[3:3 + health_bar, -2, :3] = (128, 255, 128)
            obs[3:3 + shooting_bar, -1, :3] = (128, 128, 255)


        # show general off-screen location
        if (observer is not None) and (observer.team == self.TEAM_BLUE or self.scenario.general_always_visible):

            dx = self.general_location[0] - observer.x
            dy = self.general_location[1] - observer.y

            if abs(dx) > self.scenario.player_view_distance or abs(dy) > self.scenario.player_view_distance:
                dx += self.scenario.player_view_distance
                dy += self.scenario.player_view_distance
                dx = min(max(dx, 0), self.scenario.player_view_distance * 2)
                dy = min(max(dy, 0), self.scenario.player_view_distance * 2)
                self.draw_tile(obs, dx + 1, dy + 1, self.GENERAL_COLOR)

        if observer_id >= 0:
            assert obs.shape == self.observation_space.shape, \
                f"Invalid observation crop, found {obs.shape} expected {self.observation_space.shape}."

        return obs

    @property
    def general_tiles_from_edge(self):
        x,y = self.general_location
        return min(
            x, y, self.scenario.map_width - x - 1, self.scenario.map_height - y - 1
        )

    def reset(self):
        """
        Reset game.
        :return: observations
        """

        # general location
        self.general_location = (np.random.randint(3, self.scenario.map_width - 5), np.random.randint(3, self.scenario.map_height - 5))
        self.general_health = self.scenario.general_initial_health
        self.general_closest_tiles_from_edge = self.general_tiles_from_edge
        self.blue_rewards_for_winning = 10

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
        self.stats_actions *= 0
        self.stats_outcome = ""

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
            player.shooting_timeout = 0

        return self._get_observations()

    @property
    def n_players(self):
        return sum(self.scenario.player_counts)

    @property
    def living_players(self):
        return [player for player in self.players if not player.is_dead]

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