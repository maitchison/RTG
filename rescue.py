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

from marl_env import MultiAgentEnv

CELL_SIZE = 3
SIN_CHANNELS = 10 # 10 channels gets a frequencies of 2^5, which suits maps around 32 tiles in width/height
DAMAGE_PER_SHOT = 5 # this means it takes 2 shots to kill

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

SHOOT_ACTIONS = {ACTION_SHOOT_UP, ACTION_SHOOT_DOWN, ACTION_SHOOT_LEFT, ACTION_SHOOT_RIGHT}
SIGNAL_ACTIONS = {ACTION_SIGNAL_UP, ACTION_SIGNAL_DOWN, ACTION_SIGNAL_LEFT, ACTION_SIGNAL_RIGHT}
MOVE_ACTIONS = {ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT}

class RescueTheGeneralScenario():

    SCENARIOS = {
        "full": {
            "description": "This is the default scenario.",
            "map_width": 48,
            "map_height": 48
        },

        "medium": {
            "description": "A smaller version of default scenario.",
            "team_counts": (2, 2, 2),
            "map_width": 32,
            "map_height": 32
        },

        "blue4": {
            "description": "four blue two red, no green, smaller map",
            "map_width": 32,
            "map_height": 32,
            "team_counts": (2, 0, 4),
        },

        "r2g2": {
            "description": "Two red players and two green players on a medium map",
            "map_width": 48,
            "map_height": 48,
            "team_counts": (2, 2, 0),
            "n_trees": 10,
            "randomize_ids": False,             # makes life simpler, see if sepecialization develops
            "reward_per_tree": 1,
            "hidden_roles": "none",
            "max_view_distance": 5,             # makes thins a bit faster
            "team_view_distance": (5, 5, 5),    # no bonus vision for red
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",     # random start locations
            "team_shoot_timeout": (5, 5, 5)      # green is much slower at shooting
        },

        "r2g2_rp": {
            "description": "Smaller version of r2g2 with randomized ids, used for testing role prediction",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 2, 0),
            "n_trees": 10,
            "randomize_ids": True,
            "reward_per_tree": 1,
            "hidden_roles": "none",
            "max_view_distance": 5,  # makes thins a bit faster
            "team_view_distance": (5, 5, 5),  # no bonus vision for red
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",  # random start locations
            "team_shoot_timeout": (5, 5, 5)  # green is much slower at shooting
        },

        "r2g2_hrp": {
            "description": "Smaller version of r2g2 with randomized ids, used for testing role prediction with hidden roles",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 2, 0),
            "n_trees": 10,
            "randomize_ids": True,
            "reward_per_tree": 1,
            "hidden_roles": "default",
            "max_view_distance": 5,  # makes thins a bit faster
            "team_view_distance": (5, 5, 5),  # no bonus vision for red
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",  # random start locations
            "team_shoot_timeout": (5, 5, 5)  # green is much slower at shooting
        },

        "r2g3": {
            "description": "Two red players and three green players on a medium map",
            "map_width": 48,
            "map_height": 48,
            "team_counts": (2, 3, 0),
            "n_trees": 10,
            "randomize_ids": False,  # makes life simpler, see if sepecialization develops
            "reward_per_tree": 1,
            "hidden_roles": "none",
            "max_view_distance": 5,  # makes thins a bit faster
            "team_view_distance": (5, 5, 5),  # no bonus vision for red
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",  # random start locations
            "team_shoot_timeout": (5, 5, 5)  # green is much slower at shooting
        },

        "r2g4": {
            "description": "Two red players and two green players on a medium map",
            "map_width": 48,
            "map_height": 48,
            "team_counts": (2, 4, 0),
            "n_trees": 10,
            "randomize_ids": False,  # makes life simpler, see if sepecialization develops
            "reward_per_tree": 1,
            "hidden_roles": "none",
            "max_view_distance": 5,  # makes thins a bit faster
            "team_view_distance": (5, 5, 5),  # no bonus vision for red
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",  # random start locations
            "team_shoot_timeout": (5, 5, 5)  # green is much slower at shooting
        },

        "red2": {
            "description": "Two red players must find and kill general on small map.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "n_trees": 10,
            "reward_per_tree": 1,
        },

        "green2": {
            "description": "Two green players must harvest trees uncontested on a small map.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (0, 2, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "n_trees": 10,
            "reward_per_tree": 1,
        },

        "blue2": {
            "description": "Two blue players must rescue the general on a small map.",
            "map_width": 16,
            "map_height": 16, # smaller to make it easier
            "team_counts": (0, 0, 2),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "n_trees": 10,
            "reward_per_tree": 1,
            "timeout_mean": 1000
        },

        "mem0": {
            "description": "A test to make sure memory works.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "bonus_actions": True,
            "bonus_actions_delay": 0,
            "timeout_mean": 200,
            "player_initial_health": 9999,
            "general_initial_health": 9999, # game won't end until timeout
        },

        "mem1": {
            "description": "A test to make sure memory works.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "bonus_actions": True,
            "bonus_actions_delay": 1,
            "timeout_mean": 200,
            "player_initial_health": 9999,
            "general_initial_health": 9999,  # game won't end until timeout
        },

        "mem10": {
            "description": "A test to make sure memory works.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "bonus_actions": True,
            "bonus_actions_delay": 10,
            "timeout_mean": 200,
            "player_initial_health": 9999,
            "general_initial_health": 9999,  # game won't end until timeout
        },

        # the idea here is to try and learn the other players identity
        "royale": {
            "description": "Red vs Blue, two soldiers each, in hidden roles battle royale.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 2),
            "n_trees": 0,
            "hidden_roles": "all",
            "battle_royale": True,
            "reveal_team_on_death": True
        }
    }

    def __init__(self, scenario_name=None, **kwargs):

        # defaults
        self.n_trees = 20
        self.reward_per_tree = 0.5
        self.map_width = 48
        self.map_height = 48

        self.max_view_distance = 7      # distance used for size of observational space, unused tiles are blanked out
        self.team_view_distance = (7, 5, 5)
        self.team_shoot_range = (4, 0, 0)
        self.team_counts = (4, 4, 4)
        self.team_shoot_timeout = (3, 3, 3)  # number of turns between shooting

        self.timeout_mean = 500
        self.timeout_sigma = 50       # this helps make sure games are not always in sync, which can happen if lots of
                                    # games timeout.
        self.general_always_visible = False
        self.general_initial_health = 10
        self.player_initial_health = 10
        self.location_encoding = "abs"  # none | sin | abs
        self.battle_royale = False   # removes general from game, and adds kill rewards
        self.bonus_actions = False   # provides small reward for taking an action that is indicated on agents local
                                     # observation some time after the signal appeared
        self.bonus_actions_delay = 10
        self.enable_signals = False
        self.starting_locations = "together"
        self.randomize_ids = True # randomize the starting ID colors each reset
        # enables team colors on agents local observation. This can be useful if one policy plays all 3 teams,
        # however it could cause problems if you want to infer what a different team would have done in that situation
        self.local_team_colors = False

        # default is red knows red, but all others are hidden
        # all is all roles are hidden
        # none is all roles are visible

        self.hidden_roles = "default"

        self.reveal_team_on_death = False

        self.description = "The full game"

        # scenario settings
        settings_to_apply = {}
        if scenario_name is not None:
            settings_to_apply = self.SCENARIOS[scenario_name].copy()

        settings_to_apply.update(**kwargs)

        # apply settings
        for k,v in settings_to_apply.items():
            assert hasattr(self, k), f"Invalid scenario attribute {k}"
            setattr(self, k, v)

    def __str__(self):
        result = []
        for k,v in vars(self).items():
            result.append(f"{k:<24} = {v}")
        return "\n".join(result)


class RTG_Player():

    def __init__(self, index, scenario: RescueTheGeneralScenario):
        # the player's index, this is fixed and used to index into the players array, i.e. self.players[id]
        self.index = index
        # this is the id number used to identify which player this is. These id values are randomized each round
        self.public_id = 0
        # position of player
        self.x, self.y = int(), int()

        self.health = int()
        self.team = int()
        self.action = int()
        self.turns_until_we_can_shoot = int()
        self.scenario = scenario
        self.was_hit_this_round = False
        self.custom_data = dict()

    @property
    def team_color(self):
        return RescueTheGeneralEnv.TEAM_COLOR[self.team]

    @property
    def id_color(self):
        return RescueTheGeneralEnv.ID_COLOR[self.public_id]

    @property
    def is_dead(self):
        return self.health <= 0

    @property
    def is_alive(self):
        return not self.is_dead

    @property
    def view_distance(self):
        return self.scenario.team_view_distance[self.team]

    @property
    def shoot_range(self):
        return self.scenario.team_shoot_range[self.team]

    @property
    def shooting_timeout(self):
        return self.scenario.team_shoot_timeout[self.team]

    @property
    def can_shoot(self):
        return self.shoot_range > 0 and self.turns_until_we_can_shoot <= 0

    def update(self):
        # update shooting cooldown
        if self.shoot_range > 0:
            self.turns_until_we_can_shoot = max(0, self.turns_until_we_can_shoot - 1)

    def in_vision(self, x, y):
        """
        Returns if given co-ordinates are within vision of the given player or not.
        """
        return max(abs(self.x - x), abs(self.y - y)) <= self.view_distance

    def damage(self, damage):
        """
        Causes player to receive given damage.
        :param damage:
        :return:
        """
        self.health = max(0, self.health - damage)
        self.was_hit_this_round = True

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
    * If game times out then any green players still alive are awarded 5 points


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
    NEUTRAL_COLOR = np.asarray([96, 96, 96], dtype=np.uint8)
    # this grass color visually indicates where edges of map are
    # I'm going to try using back though to see if it helps with observation predictions
    # GRASS_COLOR = np.asarray([24, 42, 16], dtype=np.uint8)
    GRASS_COLOR = np.asarray([0, 0, 0], dtype=np.uint8)
    TREE_COLOR = np.asarray([12, 174, 91], dtype=np.uint8)
    DEAD_COLOR = np.asarray([0, 0, 0], dtype=np.uint8)

    ID_COLOR = np.asarray(
        [np.asarray(plt.cm.get_cmap("tab20")(i)[:3])*255 for i in range(20)]
    , dtype=np.uint8)

    TEAM_COLOR = np.asarray([
        (255,25,25),
        (25,255,25),
        (25,25,255)
    ], dtype=np.uint8)

    DX = [0, 0, -1, +1]
    DY = [-1, 1, 0, 0]

    get_current_epoch = None  # assign a function to this that returns current epoch

    def __init__(self, scenario_name:str= "full", name:str= "env", log_file:str=None, dummy_prob=0, **scenario_kwargs):
        """
        :param scenario_name:
        :param name:
        :param log_file:
        :param dummy_prob: Probability (0..1) that a player will be removed from game. This allows for a random number of
            players in the game.
        :param scenario_kwargs:
        """

        # setup our scenario
        self.scenario = RescueTheGeneralScenario(scenario_name, **scenario_kwargs)

        super().__init__(sum(self.scenario.team_counts))

        self.env_create_time = time.time()

        self.log_file = log_file
        self._needs_repaint = True

        self.action_space = gym.spaces.Discrete(14 if self.scenario.enable_signals else 10)

        self.name = name
        self.counter = 0
        self.timeout = 0
        self.game_counter = 0
        self.dummy_prob = dummy_prob

        self.general_location = (0,0)
        self.general_health = int()
        self.general_closest_tiles_from_edge = int()
        self.blue_has_stood_next_to_general = bool()
        self.blue_rewards_for_winning = int()

        # create players and assign teams
        self.players = [RTG_Player(index, self.scenario) for index in range(self.n_players)]
        teams = [self.TEAM_RED] * self.scenario.team_counts[0] + \
                [self.TEAM_GREEN] * self.scenario.team_counts[1] + \
                [self.TEAM_BLUE] * self.scenario.team_counts[2]

        for index, team in enumerate(teams):
            self.players[index].team = team

        self.team_scores = np.zeros([3], dtype=np.float)

        self.previous_team_scores = np.zeros([3], dtype=np.float)
        self.previous_outcome = str()

        # create map, and a lookup (just for optimization
        self.map = np.zeros((self.scenario.map_width, self.scenario.map_height), dtype=np.int)
        self.player_lookup = np.zeros((self.scenario.map_width, self.scenario.map_height), dtype=np.int)

        self.stats_player_hit = np.zeros((3,3), dtype=np.int) # which teams killed who
        self.stats_deaths = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_kills = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_general_shot = np.zeros((3,), dtype=np.int)  # which teams shot general
        self.stats_general_moved = np.zeros((3,), dtype=np.int)  # which teams moved general
        self.stats_general_hidden = np.zeros((3,), dtype=np.int)  # which teams stood ontop of general
        self.stats_tree_harvested = np.zeros((3,), dtype=np.int)  # which teams harvested trees


        self.stats_actions = np.zeros((3, self.action_space.n), dtype=np.int)
        self.outcome = str() # outcome of game

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
            shape=((self.scenario.max_view_distance * 2 + 3) * CELL_SIZE, (self.scenario.max_view_distance * 2 + 3) * CELL_SIZE, obs_channels),
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
            index = self.player_lookup[x, y]

            player = self.players[index] if index >= 0 else None

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
            self.player_lookup[new_x, new_y] = player.index
            player.x = new_x
            player.y = new_y

    def step(self, actions):
        """
        Perform game step
        :param actions: np array of actions of dims [n_players]
        :return: observations, rewards, dones, infos
        """
        assert self.game_counter > 0, "Must call reset before step."
        assert len(actions) == self.n_players, f"{self.name}: Invalid number of players"
        assert self.outcome == "", f"{self.name}: Game has concluded with result {self.outcome}, reset must be called."

        green_tree_harvest_counter = 0

        rewards = np.zeros([self.n_players], dtype=np.float)
        dones = np.zeros([self.n_players], dtype=np.bool)
        infos = [{} for _ in range(self.n_players)]

        # assign actions to players / remove invalid actions
        for action, player, info in zip(actions, self.players, infos):

            # if player was dead at start of round ignore this transition when training
            if player.is_dead:
                info["train_mask"] = 0

            player.action = ACTION_NOOP if player.is_dead else action

            if player.action in SHOOT_ACTIONS and not player.can_shoot:
                player.action = ACTION_NOOP

        # count actions
        for player in self.players:
            self.stats_actions[player.team, player.action] += 1

        # count predictions (if given)
        # todo... (requires player.prediction to be set)
        # for player in self.living_players:
        #     for target_player in self.players:
        #         if player.prediction[target_player.id] == target_player.team:
        #             self.stats_predictions_correct[player.team] += 1
        #         else:
        #             self.stats_predictions_wrong[player.team] += 1

        # apply actions, we process actions in the following order..
        # shooting
        # moving + acting

        red_team_good_kills = 0
        blue_team_good_kills = 0
        team_self_kills = [0, 0, 0]
        team_deaths = [0, 0, 0]

        for player in self.players:
            player.was_hit_this_round = False

        # -----------------------------------------
        # shooting
        # note living players will give us all players living before the shooting starts, so players killed during
        # combat still get to shoot this round
        for player in self.living_players:

            if player.action not in SHOOT_ACTIONS:
                continue

            index = player.action - ACTION_SHOOT_UP
            x = player.x
            y = player.y

            player.turns_until_we_can_shoot = player.shooting_timeout

            for j in range(player.shoot_range):
                # check location

                x += self.DX[index]
                y += self.DY[index]

                if x < 0 or x >= self.scenario.map_width or y < 0 or y >= self.scenario.map_height:
                    break

                # check other players
                other_player = self.player_at_pos(x, y)
                if other_player is not None:
                    # a soldier was hit...
                    other_player.damage(DAMAGE_PER_SHOT)
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
                    self.general_health -= DAMAGE_PER_SHOT
                    self.stats_general_shot[player.team] += 1
                    self._needs_repaint = True
                    break

        # perform update
        for player in self.players:
            player.update()

        # -------------------------
        # action button
        general_has_been_moved = False
        general_is_closer_to_edge = False
        for player in self.living_players:

            if player.action != ACTION_ACT:
                continue

            # harvest a tree if we are standing on it
            if self.map[(player.x, player.y)] == self.MAP_TREE:
                self.stats_tree_harvested[player.team] += 1
                if player.team == self.TEAM_GREEN:
                    green_tree_harvest_counter += 1
                self.map[(player.x, player.y)] = self.MAP_GRASS
                self._needs_repaint = True
                continue

            if general_has_been_moved:
                continue

            # move general by one tile if we are standing next to them
            player_distance_from_general = abs(player.x - self.general_location[0]) + abs(player.y - self.general_location[1])

            if player_distance_from_general == 1:
                previous_general_location = self.general_location
                self.general_location = (player.x, player.y)
                # moving the general is a once per turn thing
                general_has_been_moved = True
                self._needs_repaint = True

                self.stats_general_moved[player.team] += 1

                # award some score if general is closer to the edge than they used to be
                if self.general_tiles_from_edge < self.general_closest_tiles_from_edge:
                    self.general_closest_tiles_from_edge = self.general_tiles_from_edge
                    general_is_closer_to_edge = True

        # ------------------------
        # moving
        for player in self.living_players:
            if player.action in MOVE_ACTIONS:
                index = player.action - ACTION_MOVE_UP
                self.move_player(player, self.DX[index], self.DY[index])

        # look for players standing on general
        for player in self.players:
            if player.x == self.general_location[0] and player.y == self.general_location[1]:
                self.stats_general_hidden[player.team] += 1

        # ------------------------
        # generate team rewards and look for outcomes

        result_general_killed = self.general_health <= 0
        result_general_rescued = self.general_tiles_from_edge == 0
        result_game_timeout = self.counter >= (self.timeout-1)
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
            small_reward = self.blue_rewards_for_winning / 20 # give 5% of remaining points
            team_rewards[self.TEAM_BLUE] += small_reward
            self.blue_rewards_for_winning -= small_reward # make sure blue always gets the same number of points for winning

        blue_player_standing_next_to_general = False
        for player in self.living_players:
            if player.team == self.TEAM_BLUE:
                if abs(player.x - self.general_location[0]) + abs(player.y - self.general_location[1]) == 1:
                    blue_player_standing_next_to_general = True

        if blue_player_standing_next_to_general and not self.blue_has_stood_next_to_general:
            # very small bonus for standing next to general for the first time, might remove this later?
            # or have it as an option maybe. I think it's needed for fast training on blue2 scenario
            small_reward = 1
            team_rewards[self.TEAM_BLUE] += small_reward
            self.blue_rewards_for_winning -= small_reward  # make sure blue always gets the same number of points for winning
            self.blue_has_stood_next_to_general = True

        if self.scenario.bonus_actions and self.counter > 0:
            # reward agents for pressing actions that were indicated to them
            # -1 is because there is a natural delay of 1 between giving hints and the score here.
            expected_action = self.bonus_actions[self.counter-1]
            for index, action in enumerate(actions):
                if action == expected_action:
                    rewards[index] += 10 / self.scenario.timeout_mean # must be mean otherwise rewards are stocastic

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

        # ----------------------------------------
        # assign team rewards

        for player in self.players:
            rewards[player.index] += team_rewards[player.team]

        self.team_scores += team_rewards

        # send done notifications to players who are dead
        # note: it's better not to do this, just return done all at once, but zero out the updates
        # on players that are dead
        #for player in self.players:
        #   dones[player.index] = player.is_dead

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
                self.outcome = "general_killed"
            elif result_general_rescued:
                self.outcome = "general_rescued"
            elif result_game_timeout:
                self.outcome = "timeout"
            elif result_all_players_dead:
                self.outcome = "all_players_dead"
            elif result_red_victory:
                self.outcome = "red_win" # royale wins
            elif result_blue_victory:
                self.outcome = "blue_win"
            elif result_green_victory:
                self.outcome = "green_win"
            else:
                # general end of game tag, this shouldn't happen
                self.outcome = "complete"

            self.write_stats_to_log()
            dones[:] = True

            #print(f"{self.name}: round finished at step {self.counter}", self.team_scores, rewards)

            for info in infos:
                # record the outcome in infos as it will be lost if environment is auto reset.
                info["outcome"] = self.outcome

        obs = self._get_observations()

        self.counter += 1

        rewards *= self.REWARD_SCALE

        obs = np.asarray(obs)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        return obs, rewards, dones, infos

    def _get_observations(self):
        return [self._get_player_observation(player_id) for player_id in range(self.n_players)]

    def draw_tile(self, obs, x, y, c):
        dx, dy = x*CELL_SIZE, y*CELL_SIZE
        obs[dx:dx+CELL_SIZE, dy:dy+CELL_SIZE, :3] = c

    def can_see_role(self, observer: RTG_Player, player: RTG_Player):
        """ Returns if player_a can see player_b's role """

        if observer is None:
            # global observer
            return True

        if self.scenario.reveal_team_on_death and player.is_dead:
            return True

        if self.scenario.hidden_roles == "all":
            return observer == player
        elif self.scenario.hidden_roles == "none":
            return True
        elif self.scenario.hidden_roles == "default":
            if observer.team == self.TEAM_RED:
                return True
            else:
                return observer == player
        else:
            raise Exception(f"Invalid hidden role setting {self.scenario.hidden_roles}, use [all|none|default]")

    def _draw_soldier(self, obs: np.ndarray, player: RTG_Player, team_colors=False, highlight=False, padding=(0, 0)):
        """
        Draw soldier
        :param obs:
        :param player:
        :param team_colors: if true solider will have team colors, otherwise will draw drawn gray
        :param highlight:
        :param padding:
        :return:
        """

        if player.is_dead:
            if team_colors:
                ring_color = (self.TEAM_COLOR[player.team] // 3 + self.DEAD_COLOR // 2)
            else:
                ring_color = self.DEAD_COLOR
        else:
            ring_color = self.TEAM_COLOR[player.team] if team_colors else self.NEUTRAL_COLOR

        if highlight:
            ring_color = ring_color//2 + self.HIGHLIGHT_COLOR //2

        inner_color = player.id_color

        draw_x, draw_y = (player.x+padding[0]) * CELL_SIZE + 1, (player.y+padding[1]) * CELL_SIZE + 1

        obs[draw_x - 1:draw_x + 2, draw_y - 1:draw_y + 2, :3] = ring_color
        obs[draw_x, draw_y, :3] = inner_color

        if player.action in SHOOT_ACTIONS:
            index = player.action - ACTION_SHOOT_UP
            dx, dy = self.DX[index], self.DY[index]
            obs[draw_x + dx, draw_y + dy, :3] = self.FIRE_COLOR

        if self.scenario.enable_signals and player.action in SIGNAL_ACTIONS:
            index = player.action - ACTION_SIGNAL_UP
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
        self._map_padding = (self.scenario.max_view_distance + 1) # +1 for border
        padding = (self._map_padding*CELL_SIZE, self._map_padding*CELL_SIZE)
        obs = np.pad(obs, (padding, padding, (0, 0)), mode="constant")

        self._map_cache = obs
        self._needs_repaint = False
        return self._map_cache.copy()

    def write_stats_to_log(self):

        if self.log_file is None:
            return

        stats = [
            self.stats_player_hit,
            self.stats_deaths,
            self.stats_kills,
            self.stats_general_shot,
            self.stats_general_moved,
            self.stats_general_hidden,
            self.stats_tree_harvested,
            self.stats_actions,
            np.asarray(self.scenario.team_counts, dtype=np.int)
        ]

        def nice_print(x):
            return " ".join(str(i) for i in x.reshape(-1))

        time_since_env_started = time.time() - self.env_create_time

        epoch = "-1" if RescueTheGeneralEnv.get_current_epoch is None else RescueTheGeneralEnv.get_current_epoch()

        output_string = ",".join(
            str(x) for x in [
                self.name, epoch, self.game_counter, self.counter, *self.team_scores,
                *(nice_print(x) for x in stats),
                self.outcome, time_since_env_started, time.time()
            ]
        )

        if self.log_file not in LOGS:
            LOGS[self.log_file] = RTG_Log(self.log_file)
        LOGS[self.log_file].log(output_string)

    def _get_player_observation(self, observer_id):
        """
        Generates a copy of the map with local observations given given player.
        :param observer_id: player's perspective, -1 is global view
        :return:
        """

        # note: this is by far the slowest part of the environment, it might be worth moving this script over
        # cython and optimizing it a bit.
        # as it stands we get around 10k FPS on this environment (where 1 FPS means one step PER AGENT per second)
        # if we ignore the get_player_observations the environment runs at a healthy 70k.
        # either way it seems to be the model that slows us down in the end anyway, as training only runs at ~3k FPS

        obs = self._get_map()

        observer = self.players[observer_id] if (observer_id != -1) else None

        # paint soldiers, living over dead
        for player in self.players:
            player_is_visible = (observer is None) or observer.in_vision(player.x, player.y)
            team_colors = self.can_see_role(observer, player)

            if player.is_dead and player_is_visible:
                self._draw_soldier(
                    obs,
                    player,
                    highlight=player.was_hit_this_round,
                    team_colors=team_colors,
                    padding=(self._map_padding, self._map_padding)
                )

        for player in self.players:
            player_is_visible = (observer is None) or observer.in_vision(player.x, player.y)
            team_colors = self.can_see_role(observer, player)
            if player.index == observer_id and not self.scenario.local_team_colors:
                team_colors = False

            if not player.is_dead and player_is_visible:
                self._draw_soldier(
                    obs,
                    player,
                    highlight=player.was_hit_this_round,
                    team_colors=team_colors,
                    padding=(self._map_padding, self._map_padding)
                )

        # in the global view we paint general on-top of soldiers so we always know where he is
        if observer_id == -1:
            self._draw_general(obs, (self._map_padding, self._map_padding))

        # ego centric view
        if observer_id >= 0:
            # get our local view
            left = (self._map_padding + observer.x - (self.scenario.max_view_distance + 1)) * CELL_SIZE
            right = (self._map_padding + observer.x + (self.scenario.max_view_distance + 2)) * CELL_SIZE
            top = (self._map_padding + observer.y - (self.scenario.max_view_distance + 1)) * CELL_SIZE
            bottom = (self._map_padding + observer.y + (self.scenario.max_view_distance + 2)) * CELL_SIZE
            obs = obs[left:right, top:bottom, :]
        else:
            # just remove padding
            padding = self._map_padding * CELL_SIZE
            obs = obs[padding:-padding, padding:-padding, :]

        if observer_id >= 0:
            # blank out edges of frame, always have 1 tile due to status and indicators
            cells_to_blank_out = 1 + self.scenario.max_view_distance - observer.view_distance

            def blank_edges(obs, pixels_to_blank_out, color):
                obs[:pixels_to_blank_out, :, :3] = color
                obs[-pixels_to_blank_out:, :, :3] = color
                obs[:, :pixels_to_blank_out, :3] = color
                obs[:, -pixels_to_blank_out:, :3] = color

            blank_edges(obs, cells_to_blank_out * CELL_SIZE, [32, 32, 32])
            if self.scenario.local_team_colors:
                blank_edges(obs, 3, observer.team_color // 2)
            else:
                blank_edges(obs, 3, [128, 128, 128])
            obs[3:-3, :3, :3] = observer.id_color

            blank_edges(obs, 1, 0)

            # bars for time, health, and shooting timeout
            frame_width, frame_height, _ = obs[3:-3, 3:-3, :].shape

            time_bar = int((self.timeout - self.counter) / self.timeout * frame_width)
            obs[3:3 + time_bar, -3, :3] = (255, 255, 128)

            health_bar = int(observer.health / self.scenario.player_initial_health * frame_width)
            obs[3:3 + health_bar, -2, :3] = (128, 255, 128)

            # this isn't helpful right now and it might be difficult for agents to predict.
            # show_shooting_timeout = False
            # if show_shooting_timeout and observer.shoot_range > 0:
            #     shooting_bar = int(observer.turns_until_we_can_shoot / observer.shooting_timeout * frame_width)
            #     obs[3:3 + shooting_bar, -1, :3] = (128, 128, 255)

            # if needed add a hint for 'bonus actions'
            if self.scenario.bonus_actions:
                action_hint = self.bonus_actions[self.counter+self.scenario.bonus_actions_delay]
                obs[action_hint*3:(action_hint+1)*3, 0:3, :3] = (0, 255, 255)

        # show general off-screen location
        if (observer is not None) and (observer.team == self.TEAM_BLUE or self.scenario.general_always_visible):

            dx = self.general_location[0] - observer.x
            dy = self.general_location[1] - observer.y

            if abs(dx) > observer.view_distance or abs(dy) > observer.view_distance:
                dx += observer.view_distance
                dy += observer.view_distance
                dx = min(max(dx, 0), observer.view_distance * 2)
                dy = min(max(dy, 0), observer.view_distance * 2)
                self.draw_tile(obs, dx + 1, dy + 1, self.GENERAL_COLOR)

        if observer_id >= 0:
            assert obs.shape == self.observation_space.shape, \
                f"Invalid observation crop, found {obs.shape} expected {self.observation_space.shape}."

        return obs

    @property
    def general_tiles_from_edge(self):
        x,y = self.general_location
        return min(
            x, y, (self.scenario.map_width - 1) - x, (self.scenario.map_height - 1) - y
        )

    def reset(self):
        """
        Reset game.
        :return: observations
        """

        # save previous result so we we reset we still have this info.
        self.previous_team_scores = self.team_scores.copy()
        self.previous_outcome = self.outcome

        # general location
        self.general_location = (np.random.randint(3, self.scenario.map_width - 2), np.random.randint(3, self.scenario.map_height - 2))
        self.general_health = self.scenario.general_initial_health
        self.general_closest_tiles_from_edge = self.general_tiles_from_edge
        self.blue_has_stood_next_to_general = False
        self.blue_rewards_for_winning = 10

        self.timeout = np.random.normal(self.scenario.timeout_mean, self.scenario.timeout_sigma)

        self._needs_repaint = True

        # bonus actions
        self.bonus_actions = np.random.randint(
            low=0,
            high=self.action_space.n,
            size=[self.timeout + self.scenario.bonus_actions_delay + 1]
        )

        # create map
        self.map[:, :] = 1
        self.player_lookup[:, :] = -1

        all_locations = list(itertools.product(range(self.scenario.map_width), range(self.scenario.map_height)))
        idxs = np.random.choice(len(all_locations), size=self.scenario.n_trees, replace=False)
        for loc in [all_locations[idx] for idx in idxs]:
            self.map[loc] = 2

        self.outcome = ""

        # reset stats
        self.stats_player_hit *= 0
        self.stats_deaths *= 0
        self.stats_kills *= 0
        self.stats_general_shot *= 0
        self.stats_general_moved *= 0
        self.stats_general_hidden *= 0
        self.stats_tree_harvested *= 0
        self.stats_actions *= 0

        # initialize players location
        if self.scenario.starting_locations == "random":

            # players are placed randomly around the map
            start_locations = [all_locations[i] for i in np.random.choice(range(len(all_locations)), size=self.n_players)]

        elif self.scenario.starting_locations == "together":

            # players are placed together but not right ontop of the general
            general_filter = lambda p: \
                abs(p[0] - self.general_location[0]) > 4 and \
                abs(p[1] - self.general_location[1]) > 4

            assert self.n_players <= 16, "This method of initializing starting locations only works with 16 or less players."

            valid_start_locations = list(filter(general_filter, all_locations))
            start_location_center = valid_start_locations[np.random.choice(range(len(valid_start_locations)))]
            start_locations = []
            for dx in range(-3, 3+1):
                for dy in range(-3, 3+1):
                    x, y = start_location_center[0]+dx, start_location_center[1]+dy
                    if 0 <= x < self.scenario.map_width:
                        if 0 <= y < self.scenario.map_height:
                            start_locations.append((x, y))
            start_locations = [start_locations[i] for i in np.random.choice(range(len(start_locations)), self.n_players)]
        else:
            raise Exception(f"Invalid starting location mode {self.scenario.starting_locations}")

        # setup the rest of the game
        self.counter = 0
        self.game_counter += 1

        self.team_scores *= 0

        ids = list(range(self.n_players))
        if self.scenario.randomize_ids:
            np.random.shuffle(ids)

        # initialize the players
        for id, player in zip(ids, self.players):

            player.public_id = id
            player.x, player.y = start_locations.pop()

            self.player_lookup[player.x, player.y] = player.index
            player.health = self.scenario.player_initial_health
            player.turns_until_we_can_shoot = player.shooting_timeout
            player.custom_data = dict()

        # apply dummy players, we can do this by simply killing them
        players_left_alive = len(self.players)
        for player in self.players:
            # make sure not to kill the last player
            if np.random.rand() < self.dummy_prob and players_left_alive > 1:
                player.health = 0
                self.player_lookup[player.x, player.y] = -1
                players_left_alive -= 1

        return np.asarray(self._get_observations())

    @property
    def n_players(self):
        return sum(self.scenario.team_counts)

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

    def _render_rgb(self, show_location=False, role_predictions=None):
        """
        Render out a frame
        :param show_location: displays location information
        :param role_predictions: (optional) np array of dims [n_players, n_players, 3] indicating predictions for each
            role as a probability distribution
        :return:
        """

        global_frame = self._process_obs(self._get_player_observation(-1), show_location)

        player_frames = [
            self._process_obs(self._get_player_observation(player_id), show_location) for player_id in range(self.n_players)
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
        elif 12 <= self.n_players <= 12:
            grid_width = 4
            grid_height = 3
        elif self.n_players <= 16:
            grid_width = 4
            grid_height = 4
        else:
            grid_width = grid_height = math.ceil(self.n_players ** 0.5)

        frame = np.zeros((gw+pw*grid_width, max(gh, ph*grid_height), 3), np.uint8)
        self._draw(frame, 0, 0, global_frame)

        # this just puts players in id order, so we can keep track of which color learns which strategies
        ids = []
        for player in self.players:
            ids.append((player.public_id, player.index))
        ids = sorted(ids)
        index_order = [index for id, index in ids]

        i = 0
        for x in range(grid_width):
            for y in range(grid_height):
                # draw a darker version of player observations so they don't distract too much
                if i < len(player_frames):
                    self._draw(frame, gw + pw*x, ph*y, player_frames[index_order[i]] * 0.75)
                i = i + 1

        # add video padding
        padding = (CELL_SIZE, CELL_SIZE)
        frame = np.pad(frame, (padding, padding, (0, 0)), mode="constant")

        # show current scores
        for team in range(3):
            length = max(0, int(self.team_scores[team] * 10))
            frame[0:100, team, team] = 50
            frame[0:length, team, team] = 255

        frame = frame.swapaxes(0, 1) # I'm using x,y, but video needs y,x

        return frame

    def render(self, mode='human', **kwargs):
        """ render environment """
        if mode == 'human':
            return self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb(**kwargs)
        else:
            raise ValueError(f"Invalid render mode {mode}")


class RTG_Log():

    def __init__(self, filename):
        self.filename = filename
        self.buffer = []
        self.last_write_time = 0

    def log(self, message):
        """
        Write message to log, and write out buffer if needed.
        :param message:
        :return:
        """
        self.buffer.append(message)
        time_since_last_log_write = time.time() - self.last_write_time
        if time_since_last_log_write > 120:
            self.write_to_disk()

    def write_to_disk(self):
        """
        Write log buffer to disk
        :return:
        """
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write("env_name, epoch, game_counter, game_length, score_red, score_green, score_blue, " +
                        "stats_player_hit, stats_deaths, stats_kills, " +
                        "stats_general_shot, stats_general_moved, stats_general_hidden, "
                        "stats_tree_harvested, stats_actions, " +
                        "player_count, result, wall_time, date_time" +
                        "\n")

        with open(self.filename, "a+") as f:
            for output_string in self.buffer:
                f.write(output_string + "\n")

        self.buffer.clear()
        self.last_write_time = time.time()

def flush_logs():
    """
    Writes all logs to disk.
    :return:
    """
    for k, v in LOGS.items():
        v.write_to_disk()

# shared log files
LOGS = dict()