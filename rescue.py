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
from scenarios import RescueTheGeneralScenario

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

class RTG_Player():

    def __init__(self, index, scenario: RescueTheGeneralScenario):
        # the player's index, this is fixed and used to index into the players array, i.e. self.players[id]
        self.index = index        
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
        return RescueTheGeneralEnv.ID_COLOR[self.index]

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

    def in_vision(self, x, y, view_distance=None):
        """
        Returns if given co-ordinates are within vision of the given player or not.
        """
        view_distance = view_distance or self.view_distance
        return max(abs(self.x - x), abs(self.y - y)) <= view_distance

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
    GRASS_COLOR = np.asarray([0, 128, 0], dtype=np.uint8)
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

    def __init__(
            self,
            scenario_name:str= "full",
            name:str= "env",
            log_file:str=None,
            dummy_prob=0,
            location_encoding="none",
            **scenario_kwargs
    ):
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
        self.round_timer = 0            # the timer tick for current round
        self.round_timeout = 0          # timeout for current round
        self.game_counter = 0           # counts the number of games played
        self.round_number = 0           # the current round we are on
        self.dummy_prob = dummy_prob

        self.general_location = (0,0)
        self.general_health = int()
        self.general_closest_tiles_from_edge = int()
        self.blue_has_stood_next_to_general = bool()
        self.blue_rewards_for_winning = int()
        self.location_encoding = location_encoding

        # create players
        self.players = [RTG_Player(index, self.scenario) for index in range(self.n_players)]

        self.round_outcome = str()  # outcome of round
        self.game_outcomes = []     # list of outcomes for each round

        # team scores for current round
        self.round_team_scores = np.zeros([3], dtype=np.float)
        # scores for each round, updated at end of round
        self.game_team_scores = np.zeros([self.scenario.rounds, 3], dtype=np.float)

        # scores from previous game
        self.previous_game_team_scores = np.zeros_like(self.game_team_scores)
        # list of outcomes for previous game
        self.previous_game_outcomes = []

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

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=((self.scenario.max_view_distance * 2 + 3) * CELL_SIZE, (self.scenario.max_view_distance * 2 + 3) * CELL_SIZE, 3),
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
        assert self.round_outcome == "", f"{self.name}: Game has concluded with result {self.round_outcome}, reset must be called."

        green_tree_harvest_counter = 0
        team_rewards = np.zeros([3], dtype=np.float)
        team_players_alive = np.zeros([3], dtype=np.int)

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

            for _ in range(player.shoot_range):
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

                        # issue points for kills
                        team_rewards[player.team] += self.scenario.points_for_kill[player.team, other_player.team]

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
        result_game_timeout = self.round_timer >= (self.timeout - 1)
        result_all_players_dead = all(player.is_dead for player in self.players)
        result_red_victory = False
        result_blue_victory = False
        result_green_victory = False

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

        if self.scenario.bonus_actions and self.round_timer > 0:
            # reward agents for pressing actions that were indicated to them
            # there is a natural delay of 1
            expected_action = self.bonus_actions[self.round_timer - 1]
            if expected_action >= 0:
                bonus_count = len(self.bonus_actions[self.bonus_actions >= 0])
                for index, action in enumerate(actions):
                    if action == expected_action:
                        # must be mean otherwise rewards are stochastic
                        # this makes rewards total to ~10
                        rewards[index] += 10 / bonus_count

        # record who is left
        living_red_players = sum(not player.is_dead for player in self.players if player.team == self.TEAM_RED)
        living_green_players = sum(not player.is_dead for player in self.players if player.team == self.TEAM_GREEN)
        living_blue_players = sum(not player.is_dead for player in self.players if player.team == self.TEAM_BLUE)

        if self.scenario.battle_royale:
            # green has the extra condition that the must finish harvesting the remaining trees
            # otherwise they are disavantages for killing any hidden red players
            if living_red_players > 0 and living_green_players == living_blue_players == 0:
                result_red_victory = True
            if living_green_players > 0 and \
                    living_red_players == living_blue_players == 0 and \
                    green_tree_harvest_counter == self.scenario.n_trees:
                result_green_victory = True
            if living_blue_players > 0 and living_red_players == living_green_players == 0:
                result_blue_victory = True
        else:
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

        self.round_team_scores += team_rewards

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
                self.round_outcome = "general_killed"
            elif result_general_rescued:
                self.round_outcome = "general_rescued"
            elif result_game_timeout:
                self.round_outcome = "timeout"
            elif result_all_players_dead:
                self.round_outcome = "all_players_dead"
            elif result_red_victory:
                self.round_outcome = "red_win" # royale wins
            elif result_blue_victory:
                self.round_outcome = "blue_win"
            elif result_green_victory:
                self.round_outcome = "green_win"
            else:
                # general end of game tag, this shouldn't happen
                self.round_outcome = "complete"

            self.write_stats_to_log()

            for info in infos:
                # record the outcome in infos as it will be lost if environment is auto reset.
                info["outcome"] = self.round_outcome

            # keep track of scores / outcomes for each round in game
            self.game_outcomes.append(self.round_outcome)
            self.game_team_scores[self.round_number] = self.round_team_scores

            # increment round, or if this was the last round send done
            if self.round_number == self.scenario.rounds-1:
                # setting dones to true will cause runner to call the hard reset
                # also send outcomes through game_info
                for info in infos:
                    info["outcomes"] = self.game_outcomes
                    info["game_scores"] = self.game_team_scores
                dones[:] = True
            else:
                # a soft reset to move us to the next round
                self.round_number += 1
                self.soft_reset()

        rewards *= self.REWARD_SCALE
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        self.round_timer += 1
        obs = self._get_observations()
        obs = np.asarray(obs)

        return obs, rewards, dones, infos

    def _get_observations(self):
        return [self._get_player_observation(player_id) for player_id in range(self.n_players)]

    def draw_tile(self, obs, x, y, c):
        dx, dy = x*CELL_SIZE, y*CELL_SIZE
        obs[dx:dx+CELL_SIZE, dy:dy+CELL_SIZE] = c

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
                inner_color = (self.TEAM_COLOR[player.team] // 3 + self.DEAD_COLOR // 2)
            else:
                inner_color = self.DEAD_COLOR
        else:
            inner_color = self.TEAM_COLOR[player.team] if team_colors else self.NEUTRAL_COLOR

        if highlight:
            inner_color = inner_color//2 + self.HIGHLIGHT_COLOR //2

        ring_color = player.id_color

        draw_x, draw_y = (player.x+padding[0]) * CELL_SIZE + 1, (player.y+padding[1]) * CELL_SIZE + 1

        obs[draw_x - 1:draw_x + 2, draw_y - 1:draw_y + 2] = inner_color
        obs[draw_x - 1:draw_x + 2, draw_y] = ring_color

        if player.action in SHOOT_ACTIONS:
            index = player.action - ACTION_SHOOT_UP
            dx, dy = self.DX[index], self.DY[index]
            obs[draw_x + dx, draw_y + dy] = self.FIRE_COLOR

        if self.scenario.enable_signals and player.action in SIGNAL_ACTIONS:
            index = player.action - ACTION_SIGNAL_UP
            dx, dy = self.DX[index], self.DY[index]
            obs[draw_x + dx, draw_y + dy] = self.SIGNAL_COLOR

    def _draw_general(self, obs, padding=(0, 0)):

        if self.scenario.battle_royale:
            return

        x, y = self.general_location
        dx, dy = (x+padding[0]) * CELL_SIZE, (y+padding[1]) * CELL_SIZE
        c = self.GENERAL_COLOR if self.general_health > 0 else self.DEAD_COLOR

        obs[dx:dx+3, dy + 1] = c
        obs[dx + 1, dy:dy+3] = c

    def _get_map(self):
        """
        Returns a map.
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
                self.name, epoch, self.game_counter, self.round_number, self.round_timer, *self.round_team_scores,
                *(nice_print(x) for x in stats),
                self.round_outcome, time_since_env_started, time.time()
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

        # paint general if they are visibile
        if observer is None or observer.in_vision(
                self.general_location[0], self.general_location[1],
                min(
                    self.scenario.team_general_view_distance[observer.team],
                    self.scenario.team_view_distance[observer.team]
                )):
            self._draw_general(obs, padding=(self._map_padding, self._map_padding))


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
            player = self.players[observer_id]
            # blank out edges of frame, always have 1 tile due to status and indicators
            cells_to_blank_out = 1 + self.scenario.max_view_distance - observer.view_distance

            def blank_edges(obs, pixels_to_blank_out, color):
                obs[:pixels_to_blank_out, :] = color
                obs[-pixels_to_blank_out:, :] = color
                obs[:, :pixels_to_blank_out] = color
                obs[:, -pixels_to_blank_out:] = color

            blank_edges(obs, cells_to_blank_out * CELL_SIZE, [32, 32, 32])
            if self.scenario.local_team_colors:
                blank_edges(obs, 3, observer.team_color // 2)
            else:
                blank_edges(obs, 3, [128, 128, 128])
            obs[3:-3, :3] = observer.id_color

            blank_edges(obs, 1, 0)

            # status lights

            status_colors = [
                (255, 255, 255), # x
                (255, 255, 255), # y
                (128, 255, 128), # health
                (255, 255, 0), # timeout
            ]

            status_values = [
                player.x / self.scenario.map_width,
                player.y / self.scenario.map_height,
                player.health / self.scenario.player_initial_health,
                self.round_timer / self.timeout
            ]

            for i, (col, value) in enumerate(zip(status_colors, status_values)):
                c = np.asarray(np.asarray(col, dtype=np.float32) * value, dtype=np.uint8)
                obs[(i+1)*3:((i+1)*3)+3, -3:-1] = c

            # if needed add a hint for 'bonus actions'
            # just for debugging
            if self.scenario.bonus_actions:
                action_hint = self.bonus_actions[self.round_timer + self.scenario.bonus_actions_delay]
                obs[action_hint*3:(action_hint+1)*3, 0:3] = (0, 255, 255)

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

    def soft_reset(self):
        """
        A soft reset rests player positions health, and the environment, but not player_teams or player colors
        :return:
        """

        # save previous result so when we reset we still have this info.
        self.previous_game_team_scores = self.round_team_scores.copy()
        self.previous_game_outcomes = self.game_outcomes[:]

        # reset game info
        self.round_outcome = ""
        self.round_timer = 0
        self.round_team_scores *= 0

        # make sure we force a repaint of the map
        self._needs_repaint = True

        # timeout is slightly random so that environments get out of sync
        self.timeout = int(
            np.random.normal(self.scenario.timeout_mean, self.scenario.timeout_mean * self.scenario.timeout_rnd))
        if self.timeout < 1:
            self.timeout = 1

        # general location
        self.general_location = (
        np.random.randint(3, self.scenario.map_width - 2), np.random.randint(3, self.scenario.map_height - 2))
        self.general_health = self.scenario.general_initial_health
        self.general_closest_tiles_from_edge = self.general_tiles_from_edge
        self.blue_has_stood_next_to_general = False
        self.blue_rewards_for_winning = 10

        # handle bonus actions
        self.bonus_actions = np.random.randint(
            low=0,
            high=self.action_space.n,
            size=[self.timeout + self.scenario.bonus_actions_delay + 1]
        )

        # zero out actions so that agent only has to remember one at a time
        if self.scenario.bonus_actions_one_at_a_time and self.scenario.bonus_actions_delay > 0:
            for i in range(len(self.bonus_actions)):
                if i % self.scenario.bonus_actions_delay != (self.scenario.bonus_actions_delay - 1):
                    self.bonus_actions[i] = -1

        # create the map
        self.map[:, :] = 1
        self.player_lookup[:, :] = -1

        all_locations = list(itertools.product(range(self.scenario.map_width), range(self.scenario.map_height)))
        idxs = np.random.choice(len(all_locations), size=self.scenario.n_trees, replace=False)
        for loc in [all_locations[idx] for idx in idxs]:
            self.map[loc] = 2

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
                    # todo, fix the +dx, +dy warning
                    x, y = start_location_center[0]+dx, start_location_center[1]+dy
                    if 0 <= x < self.scenario.map_width:
                        if 0 <= y < self.scenario.map_height:
                            start_locations.append((x, y))
            start_locations = [start_locations[i] for i in np.random.choice(range(len(start_locations)), self.n_players)]
        else:
            raise Exception(f"Invalid starting location mode {self.scenario.starting_locations}")

        # initialize the players
        for player in self.players:
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

        # reset stats
        self.stats_player_hit *= 0
        self.stats_deaths *= 0
        self.stats_kills *= 0
        self.stats_general_shot *= 0
        self.stats_general_moved *= 0
        self.stats_general_hidden *= 0
        self.stats_tree_harvested *= 0
        self.stats_actions *= 0

        return np.asarray(self._get_observations())

    def reset(self):
        """
        Perform hard reset of game.
        :return: observations
        """

        self.game_counter += 1
        self.round_number = 0

        self.game_team_scores *= 0
        self.game_outcomes = []

        # assign random teams
        teams = [self.TEAM_RED] * self.scenario.team_counts[0] + \
                [self.TEAM_GREEN] * self.scenario.team_counts[1] + \
                [self.TEAM_BLUE] * self.scenario.team_counts[2]
        np.random.shuffle(teams)
        for index, team in enumerate(teams):
            self.players[index].team = team

        return self.soft_reset()

    @property
    def n_players(self):
        return sum(self.scenario.team_counts)

    @property
    def living_players(self):
        return [player for player in self.players if not player.is_dead]

    def _render_human(self):
        raise NotImplemented("Sorry tile-map rendering not implemented yet")

    def _draw(self, frame, x, y, image):
        w,h,_ = image.shape
        frame[x:x+w, y:y+h] = image

    def _render_rgb(self):
        """
        Render out a frame
        :return:
        """

        global_frame = self._get_player_observation(-1)

        player_frames = [self._get_player_observation(player_id) for player_id in range(self.n_players)]

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

        # show current scores
        for team in range(3):
            length = max(0, int(self.round_team_scores[team] * 10))
            frame[0:100, team, team] = 50
            frame[0:length, team, team] = 255

        frame = frame.swapaxes(0, 1) # I'm using x,y, but video needs y,x

        return frame

    def render(self, mode='human'):
        """ render environment """
        if mode == 'human':
            return self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb()
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
                f.write(
                    "env_name, epoch, game_counter, round_counter, game_length, score_red, score_green, score_blue, " +
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