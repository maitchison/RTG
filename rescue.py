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

from typing import Union

from marl_env import MultiAgentEnv
from scenarios import RescueTheGeneralScenario
from utils import draw_line, draw_pixel

from typing import Tuple, List

CELL_SIZE = 3

ACTION_NOOP = 0
ACTION_MOVE_UP = 1
ACTION_MOVE_DOWN = 2
ACTION_MOVE_LEFT = 3
ACTION_MOVE_RIGHT = 4
ACTION_ACT = 5
ACTION_SHOOT_UP = 6
ACTION_SHOOT_DOWN = 7
ACTION_SHOOT_LEFT = 8
ACTION_SHOOT_RIGHT = 9

# how many actions the game has
NUM_ACTIONS = 10

SHOOT_ACTIONS = {ACTION_SHOOT_UP, ACTION_SHOOT_DOWN, ACTION_SHOOT_LEFT, ACTION_SHOOT_RIGHT}
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
        self.invisible = False
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
    def pos(self):
        return (self.x, self.y)

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
        return max_distance(*self.pos, x, y) <= view_distance

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

    MAP_GRASS = 1
    MAP_TREE = 2

    TEAM_RED = 0
    TEAM_GREEN = 1
    TEAM_BLUE = 2

    FIRE_COLOR = np.asarray([255, 255, 0], dtype=np.uint8)
    SIGNAL_COLOR = np.asarray([20, 20, 20], dtype=np.uint8)
    HIGHLIGHT_COLOR = np.asarray([180, 180, 50], dtype=np.uint8)
    GENERAL_COLOR = np.asarray([255, 255, 255], dtype=np.uint8)
    NEUTRAL_COLOR = np.asarray([96, 96, 96], dtype=np. uint8)
    GRASS_COLOR = np.asarray([16, 64, 24], dtype=np.uint8)
    TREE_COLOR = np.asarray([12, 174, 91], dtype=np.uint8)
    BUTTON_COLOR = np.asarray([200, 200, 200], dtype=np.uint8)
    DEAD_COLOR = np.asarray([0, 0, 0], dtype=np.uint8)

    ID_COLOR = np.asarray(
        # we mute the id colors a little so the team color stands out more
        [np.asarray(plt.cm.get_cmap("tab20")(i)[:3])*200 for i in range(20)]
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
            location_encoding="none",
            channels_first=True,        # observations will be returned as CHW
            **scenario_kwargs
    ):
        """
        :param scenario_name:
        :param name:
        :param log_file:
        :param scenario_kwargs:
        """

        # setup our scenario
        self.scenario = RescueTheGeneralScenario(scenario_name, **scenario_kwargs)

        super().__init__(sum(self.scenario.team_counts))

        self.env_create_time = time.time()

        self.log_file = log_file
        self._needs_repaint = True

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self.vote_timer = 0             # >0 indicates a vote is taking place
        self.current_vote = np.zeros([self.n_players], dtype=np.int) # who is voting for who (-1 = pass)
        self.who_called_vote: Union[RTG_Player, None] = None # who called the vote.
        self.name = name
        self.round_timer = 0            # the timer tick for current round
        self.round_timeout = 0          # timeout for current round
        self.game_counter = 0           # counts the number of games played
        self.round_number = 0           # the current round we are on

        self.general_location = (0, 0)
        self.general_health = int()
        self.general_closest_tiles_from_edge = int()
        self.blue_has_stood_next_to_general = bool()
        self.blue_rewards_for_winning = int()
        self.location_encoding = location_encoding
        self.channels_first = channels_first
        self.button_location = (0, 0)

        # turns until a vote can occur
        self.vote_cooldown = 0

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

        self.stats_player_hit = np.zeros((3,3), dtype=np.int) # which teams killed who
        self.stats_player_hit_with_witness = np.zeros((3, 3), dtype=np.int)  # who shoot who with a witness
        self.stats_deaths = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_kills = np.zeros((3,), dtype=np.int)  # how many players died
        self.stats_general_shot = np.zeros((3,), dtype=np.int)  # which teams shot general
        self.stats_general_moved = np.zeros((3,), dtype=np.int)  # which teams moved general
        self.stats_general_hidden = np.zeros((3,), dtype=np.int)  # which teams stood ontop of general
        self.stats_tree_harvested = np.zeros((3,), dtype=np.int)  # which teams harvested trees
        self.stats_votes = np.zeros((4,), dtype=np.int)  # number of players from this team who were killed by vote,
                        # with last entry being pass

        self.shooting_lines = []

        self.stats_actions = np.zeros((3, self.action_space.n), dtype=np.int)

        obs_height = (self.scenario.max_view_distance * 2 + 3) * CELL_SIZE
        obs_width = (self.scenario.max_view_distance * 2 + 3) * CELL_SIZE

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, obs_height, obs_width) if self.channels_first else (obs_height, obs_width, 3),
            dtype=np.uint8
        )

    def player_at_pos(self, x, y, include_dead = False):
        """
        Returns first player at given position or None if no players are on that tile
        :param x:
        :param y:
        :return:
        """

        for player in self.players:
            if player.is_dead and not include_dead:
                continue
            if player.x == x and player.y == y:
                return player

        return None

    def move_player(self, player, dx, dy):
        """
        Moves player given offset.
        Players may occupy cells with other players, but if they move outside of the map they will be clipped back in.
        """

        new_x = min(max(player.x + dx, 0), self.scenario.map_width - 1)
        new_y = min(max(player.y + dy, 0), self.scenario.map_height - 1)

        if self.player_at_pos(new_x, new_y) is None:
            # can not move on-top of other (living) players
            player.x = new_x
            player.y = new_y

    def _player_direct_shoot(self, player:RTG_Player, direction:int):
        """
        Return the player that would be hit if player fires in given cardinal direction.
        :param player:
        :param direction:
        :return:
        """

        x = player.x
        y = player.y
        for j in range(player.shoot_range):
            x += self.DX[direction]
            y += self.DY[direction]

            if x < 0 or x >= self.scenario.map_width or y < 0 or y >= self.scenario.map_height:
                break

            # check other players
            other_player = self.player_at_pos(x, y)
            if other_player is not None:
                return other_player

            # check general
            if not self.scenario.battle_royale and ((x, y) == self.general_location):
                return 'general'

        return None



    def _player_auto_shoot(self, player:RTG_Player):
        """
        Returns the player that would be hit if this player fires (via auto targeting).
        :param player: the player who is firing
        :return: target that was hit or None or 'general'
        """
        # look for a target to shoot, this will update so dead players can not be shot twice.
        potential_targets = []
        for target in self.living_players:
            # must be not us, not known to be on our team, and living
            if target == player:
                continue
            if self.can_see_role(player, target) and player.team == target.team:
                continue
            distance = max_distance(*player.pos, *target.pos)
            if distance <= player.shoot_range:
                potential_targets.append((distance, target.index, target))

        if len(potential_targets) == 0:
            general_distance = max_distance(*player.pos, *self.general_location)
            # no targets, check for general
            if not self.scenario.battle_royale and general_distance <= player.shoot_range and player.team != self.TEAM_BLUE:
                return 'general'
            return None

        # sort close to far, and then by index order
        potential_targets.sort()
        _, _, target = potential_targets[0]

        return target

    def _step_main(self, actions):

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

            player.turns_until_we_can_shoot = player.shooting_timeout

            if self.scenario.auto_shooting:
                target = self._player_auto_shoot(player)
            else:
                target = self._player_direct_shoot(player, player.action - ACTION_SHOOT_UP)

            # check who was hit
            if target is None:
                # nothing was hit, but still display shooting line
                hit_x = player.x + self.DX[player.action - ACTION_SHOOT_UP] * player.shoot_range
                hit_y = player.y + self.DY[player.action - ACTION_SHOOT_UP] * player.shoot_range
                #self.shooting_lines.append((*player.pos, hit_x, hit_y))
                continue
            elif target == 'general':
                # general was hit
                self.general_health -= self.scenario.team_shoot_damage[player.team]
                self.stats_general_shot[player.team] += 1
                self._needs_repaint = True
                #self.shooting_lines.append((*player.pos, *self.general_location))
            else:
                # target was another player, so damage them and display shooting line
                target.damage(self.scenario.team_shoot_damage[player.team])
                self.stats_player_hit[player.team, target.team] += 1
                #self.shooting_lines.append((*player.pos, *target.pos))

                # look for witnesses
                for witness in self.players:
                    if witness.is_alive and witness != player and witness.team != player.team and \
                        player.in_vision(*witness.pos):
                        self.stats_player_hit_with_witness[player.team, target.team] += 1

                if target.is_dead:
                    # we killed the target player
                    self.stats_deaths[target.team] += 1
                    self.stats_kills[player.team] += 1
                    team_deaths[target.team] += 1
                    if player.team == target.team:
                        team_self_kills[player.team] += 1
                    if player.team == self.TEAM_RED and target.team == self.TEAM_BLUE:
                        red_team_good_kills += 1
                    elif player.team == self.TEAM_BLUE and target.team == self.TEAM_RED:
                        blue_team_good_kills += 1
                    # issue points for kills
                    team_rewards[player.team] += self.scenario.points_for_kill[player.team, target.team]

        # perform update
        for player in self.players:
            player.update()

        # -------------------------
        # call vote button (combined with action button)

        if self.vote_cooldown > 0:
            self.vote_cooldown -= 1

        for player in self.living_players:

            # to call a vote we must be standing next to the button or a dead body

            if not self.scenario.enable_voting or player.action != ACTION_ACT or self.vote_cooldown > 0:
                continue

            # to call a vote player must be alive, and close to a dead body
            near_body = None
            for target in self.players:
                if target.is_dead and max_distance(*player.pos, *target.pos) <= 1 and not target.invisible:
                    near_body = target

            near_button = max_distance(*player.pos, *self.button_location) <= 1

            if near_body:
                self.vote_timer = 10
                self.vote_cooldown = 20
                self.who_called_vote = player
                # 'remove' the body by make it invisible
                near_body.invisible = True
            elif near_button:
                self.vote_timer = 10
                self.vote_cooldown = 20
                self.who_called_vote = player

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

            # disable general moving if they have been moved, or if battle royale is enabled
            if general_has_been_moved or self.scenario.battle_royale:
                continue

            # move general by one tile if we are standing next to them
            player_distance_from_general = l1_distance(*player.pos, *self.general_location)

            # make sure there are enough other players to move the general
            players_nearby = 0
            for other_player in self.players:
                if other_player.is_dead:
                    continue
                if max_distance(*player.pos, *other_player.pos) <= 1:
                    players_nearby += 1 # this includes ourself.

            if players_nearby < self.scenario.players_to_move_general:
                continue

            if player_distance_from_general == 1:
                previous_general_location = self.general_location
                self.general_location = player.pos
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

        result_general_rescued = self.general_tiles_from_edge == 0
        result_game_timeout = self.round_timer >= (self.timeout - 1)
        result_all_players_dead = all(player.is_dead for player in self.players)
        result_red_victory = False
        result_blue_victory = False
        result_green_victory = False
        red_seen_general = False

        team_rewards[self.TEAM_GREEN] += green_tree_harvest_counter * self.scenario.reward_per_tree

        for player in self.players:
            if not player.is_dead:
                team_players_alive[player.team] += 1

        for player in self.players:
            if not player.is_dead and player.team == self.TEAM_RED:
                if player.in_vision(*self.general_location, self.scenario.team_general_view_distance[self.TEAM_RED]):
                    red_seen_general = True

        result_general_killed = self.general_health <= 0 or (self.scenario.red_wins_if_sees_general and red_seen_general)

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
                if l1_distance(*player.pos, *self.general_location) == 1:
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
            # otherwise they are disadvantages for killing any hidden red players
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

        # apply zero sum rules
        if self.scenario.zero_sum:
            new_team_rewards = team_rewards.copy()
            new_team_rewards[0] -= (team_rewards[1] + team_rewards[2])
            new_team_rewards[1] -= (team_rewards[0] + team_rewards[2])
            new_team_rewards[2] -= (team_rewards[0] + team_rewards[1])
            team_rewards = new_team_rewards

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

        return rewards, dones, infos

    def _step_vote(self, actions):
        """ Logic for voting system"""

        self.vote_timer -= 1

        rewards = np.zeros([self.n_players], dtype=np.float)
        dones = np.zeros([self.n_players], dtype=np.bool)
        infos = [{} for _ in range(self.n_players)]

        for player in self.players:
            player.was_hit_this_round = False
            player.action = actions[player.index]

        # special logic for voting (don't apply actions, instead use them as a vote)
        # on step 5 the votes are locked in
        if self.vote_timer >= 5:
            for player in self.players:
                # action 0 is pass, action 1..n is that player number -1, all remaining actions are pass.
                # also, only living players can vote
                if player.is_alive and 0 <= player.action < self.n_players:
                    self.current_vote[player.index] = player.action - 1
                else:
                    self.current_vote[player.index] = -1

        team_rewards = np.zeros([3], dtype=np.float)

        # on step 10 we implement vote and end the voting session
        if self.vote_timer == 0:

            living_players = len(self.living_players)
            votes_needed = math.ceil(living_players / 2 + 0.0001)

            for target in self.players:
                votes_for_this_player = sum(self.current_vote == target.index)

                if votes_for_this_player >= votes_needed:
                    self.stats_votes[target.team] += 1

                    # execute kill, and remove player
                    target.health = 0
                    target.invisible = True

                    # issue points for kill
                    for team in range(3):
                        team_rewards[team] += self.scenario.points_for_kill[team, target.team]

                    break

            # record total number of votes
            self.stats_votes[-1] += 1

            # reset voting
            self.vote_timer = 0
            self.who_called_vote = None

        self.round_team_scores += team_rewards

        for player in self.players:
            rewards[player.index] += team_rewards[player.team]

        return rewards, dones, infos

    def step(self, actions):
        """
        Perform game step
        :param actions: np array of actions of dims [n_players]
        :return: observations, rewards, dones, infos
        """
        assert self.game_counter > 0, "Must call reset before step."
        assert len(actions) == self.n_players, f"{self.name}: Invalid number of players"
        assert self.round_outcome == "", f"{self.name}: Game has concluded with result {self.round_outcome}, reset must be called."

        self.shooting_lines = []

        if self.vote_timer > 0:
            rewards, dones, infos = self._step_vote(actions)
        else:
            rewards, dones, infos = self._step_main(actions)

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

        team_color = self.TEAM_COLOR[player.team] if team_colors else self.NEUTRAL_COLOR
        id_color = player.id_color

        if player.is_dead:
            id_color = id_color // 3
            team_color = team_color // 3

        if highlight:
            id_color = id_color // 2 + self.HIGHLIGHT_COLOR //2

        draw_x, draw_y = (player.x+padding[0]) * CELL_SIZE + 1, (player.y+padding[1]) * CELL_SIZE + 1

        obs[draw_x - 1:draw_x + 2, draw_y - 1:draw_y + 2] = id_color
        obs[draw_x, draw_y] = team_color

        # show shooting
        if player.action in SHOOT_ACTIONS:
            index = player.action - ACTION_SHOOT_UP
            dx, dy = self.DX[index], self.DY[index]
            obs[draw_x+dx, draw_y+dy] = self.FIRE_COLOR

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

        # paint button
        if self.scenario.voting_button:
            self.draw_tile(obs, *self.button_location, self.BUTTON_COLOR)

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
            self.stats_player_hit_with_witness,
            self.stats_deaths,
            self.stats_kills,
            self.stats_general_shot,
            self.stats_general_moved,
            self.stats_general_hidden,
            self.stats_tree_harvested,
            self.stats_actions,
            self.stats_votes,
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

    def _draw_voting_screen(self, obs):
        """
        Overlay the voting screen on top of observation. Voting info will be centred.

        :param obs: nd array of dims [W, H, 3] and type uint8
        :return:
        """

        W, H, _ = obs.shape

        req_width = (self.n_players + 1) * 3
        req_height = (self.n_players + 2) * 3

        dx = (W - req_width) // 2
        dy = (H - req_height) // 2

        # fade out background
        obs[3:-3, 3:-3] //= 4

        # clear area under voting info
        obs[dx-1:dx+req_width+1, dy-1:dy+req_height+1] = 0

        # show who called vote
        if self.who_called_vote is not None:
            draw_pixel(obs, dx, dy, self.who_called_vote.id_color, size=3)

        # display a matrix indicating who is voting for who
        for player in self.players:
            c = player.id_color
            # going down we have each player
            draw_pixel(obs, dx, dy + (player.index + 1) * 3, c, size=3)
            # going across we have who is being voted for
            draw_pixel(obs, dx + (player.index + 1) * 3, dy, c, size=3)
            # show which votes are valid
            for target in self.players:
                if player.is_dead or target.is_dead:
                    c = 32
                else:
                    c = 0
                draw_pixel(obs, dx + (target.index + 1) * 3, dy + (player.index + 1) * 3, c, size=3)

            # now display the vote
            vote = self.current_vote[player.index]
            if 0 <= vote < self.n_players and self.players[vote].is_alive:
                draw_pixel(obs, dx + (vote + 1) * 3, dy + (player.index + 1) * 3, [128, 128, 128], size=3)

        # display a countdown
        obs[dx:dx+int((self.vote_timer/10) * req_width), dy+req_height-3:dy+req_height, :] = [255, 255, 0]

    def _get_player_observation(self, observer_id, force_channels_last=False):
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

        # paint shooting lines
        for x1, y1, x2, y2 in self.shooting_lines:
            x1, y1 = x1 + self._map_padding, y1 + self._map_padding
            x2, y2 = x2 + self._map_padding, y2 + self._map_padding
            draw_line(obs, x1*3+1, y1*3+1, x2*3+1, y2*3+1, [255, 255, 0])

        # paint general if they are visible
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

            if player.invisible and player != observer:
                # do not show invisible bodies (ones that have been removed from game)
                player_is_visible = False

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

            if player.invisible and player != observer:
                # do not show invisible bodies (ones that have been removed from game)
                player_is_visible = False

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
            blank_edges(obs, 3, [128, 128, 128])

            # display id color
            obs[3:-3, :3] = observer.id_color

            blank_edges(obs, 1, 0)

            # status lights
            status_colors = [
                (255, 255, 255), # x
                (255, 255, 255), # y
                (128, 255, 128), # health
                (255, 255, 0),   # timeout
                (255, 255, 0),   # shooting timeout (warming up)
                (255, 255, 255),   # general distance
            ]

            general_distance = max_distance(*player.pos, *self.general_location) / max(
                self.scenario.map_width, self.scenario.map_height)

            status_values = [
                player.x / self.scenario.map_width,
                player.y / self.scenario.map_height,
                player.health / self.scenario.player_initial_health,
                self.round_timer / self.timeout,
                1-(player.turns_until_we_can_shoot / player.shooting_timeout) if player.shooting_timeout > 0 else 1,
                general_distance if player.team == self.TEAM_BLUE and self.scenario.blue_general_indicator == "distance" else 0
            ]

            # change color if agent is able to shoot
            if player.can_shoot == 0:
                status_colors[4] = (255, 0, 0) # red for able to shoot

            for i, (col, value) in enumerate(zip(status_colors, status_values)):
                c = np.asarray(np.asarray(col, dtype=np.float32) * value, dtype=np.uint8)
                obs[(i+1)*3:((i+1)*3)+3, -3:-1] = c

            # show team colors
            i = len(status_values)
            obs[(i + 1) * 3:((i + 1) * 3) + 3, -3:-1] = player.team_color

            # if needed add a hint for 'bonus actions'
            # just for debugging
            if self.scenario.bonus_actions:
                action_hint = self.bonus_actions[self.round_timer + self.scenario.bonus_actions_delay]
                obs[action_hint*3:(action_hint+1)*3, 0:3] = (0, 255, 255)

        # show general off-screen location
        if (observer is not None) and (observer.team == self.TEAM_BLUE and self.scenario.blue_general_indicator == "direction"):

            dx = self.general_location[0] - observer.x
            dy = self.general_location[1] - observer.y

            if abs(dx) > observer.view_distance or abs(dy) > observer.view_distance:
                dx += self.scenario.max_view_distance
                dy += self.scenario.max_view_distance
                dx = min(max(dx, 0), observer.view_distance * 2)
                dy = min(max(dy, 0), observer.view_distance * 2)
                self.draw_tile(obs, dx + 1, dy + 1, self.GENERAL_COLOR)

        # overlay the voting screen (if we are voting)
        if self.vote_timer > 0:
            self._draw_voting_screen(obs)

        # we render everything as channels last, but might need to convert it if channels_first is set to true.
        if self.channels_first and not force_channels_last:
            obs = obs.swapaxes(1, 2)
            obs = obs.swapaxes(0, 1)

        if observer_id >= 0 and not force_channels_last:
            assert obs.shape == self.observation_space.shape, \
                f"Invalid observation crop, found {obs.shape} expected {self.observation_space.shape}."

        # frame blanking
        # note: it would be nice if this was a deterministic function of t, player_id, game_number, game_id
        if observer is not None and np.random.rand() < self.scenario.frame_blanking and self.round_timer > 4:
            return obs * 0

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

        self.vote_timer = 0
        self.who_called_vote = None
        self.current_vote[:] = -1

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

        self.vote_cooldown = 0

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

        all_locations:List[Tuple[int, int]] = [(x,y) for (x,y) in
            itertools.product(range(self.scenario.map_width), range(self.scenario.map_height))
        ]

        idxs = np.random.choice(len(all_locations), size=self.scenario.n_trees, replace=False)
        for loc in [all_locations[idx] for idx in idxs]:
            self.map[loc] = 2

        # default button location to center of map
        button_location = (self.scenario.map_width//2, self.scenario.map_height//2)

        # initialize players location
        if self.scenario.starting_locations == "random":
            # players are placed randomly around the map
            start_locations = [all_locations[i] for i in np.random.choice(range(len(all_locations)), size=self.n_players)]
        elif self.scenario.starting_locations == "together":
            # players are placed together but not right on top of the general

            min_distance_from_general = min([2, self.scenario.map_width//4, self.scenario.map_height//4])

            general_filter = lambda p: \
                abs(p[0] - self.general_location[0]) >= min_distance_from_general and \
                abs(p[1] - self.general_location[1]) >= min_distance_from_general
            assert self.n_players <= 16, "This method of initializing starting locations only works with 16 or less players."
            valid_start_locations = list(filter(general_filter, all_locations))
            start_location_center = valid_start_locations[np.random.choice(range(len(valid_start_locations)))]
            button_location = start_location_center
            start_locations = []
            for dx in range(-2, 2+1):
                for dy in range(-2, 2+1):
                    x, y = start_location_center[0]+dx, start_location_center[1]+dy
                    if 0 <= x < self.scenario.map_width:
                        if 0 <= y < self.scenario.map_height:
                            start_locations.append((x, y))
            start_locations = [start_locations[i] for i in np.random.choice(range(len(start_locations)), self.n_players)]
        else:
            raise Exception(f"Invalid starting location mode {self.scenario.starting_locations}")

        # create a button if needed
        self.button_location = button_location

        # initialize the players
        for player in self.players:
            player.x, player.y = start_locations.pop()
            player.health = self.scenario.player_initial_health
            player.turns_until_we_can_shoot = player.shooting_timeout
            player.custom_data = dict()
            player.invisible = False

        # randomly kill a few players
        players_to_kill = self.scenario.initial_random_kills
        while players_to_kill > np.random.rand():
            living_player_ids = [player.index for player in self.living_players]
            if len(living_player_ids) == 0:
                print("Warning, removed all players during initialization!")
                break
            idx = np.random.choice(living_player_ids)
            player_to_kill = self.players[idx]
            player_to_kill.health = 0
            player_to_kill.invisible = True
            players_to_kill -= 1

        # reset stats
        self.stats_player_hit *= 0
        self.stats_player_hit_with_witness *= 0
        self.stats_deaths *= 0
        self.stats_kills *= 0
        self.stats_general_shot *= 0
        self.stats_general_moved *= 0
        self.stats_general_hidden *= 0
        self.stats_tree_harvested *= 0
        self.stats_actions *= 0
        self.stats_votes *= 0

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

        global_frame = self._get_player_observation(-1, force_channels_last=True)

        player_frames = [self._get_player_observation(player_id, force_channels_last=True) for player_id in range(self.n_players)]

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
            frame[0:100, team, team] = 128
            if self.round_team_scores[team] >= 0:
                length = max(0, int(self.round_team_scores[team] * 10))
                frame[0:length, team, team] = 255
            else:
                length = max(0, -int(self.round_team_scores[team] * 10))
                frame[0:length, team, team] = 32

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
                    "stats_player_hit, stats_player_hit_with_witness, stats_deaths, stats_kills, " +
                    "stats_general_shot, stats_general_moved, stats_general_hidden, "
                    "stats_tree_harvested, stats_actions, " +
                    "stats_votes, " +
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

def l1_distance(x1, y1, x2, y2):
    """ Returns l1 (manhattan) distance."""
    return abs(x1-x2) + abs(y1-y2)

def max_distance(x1, y1, x2, y2):
    """ Returns max(abs(dx), abs(dy))."""
    return max(abs(x1-x2), abs(y1-y2))

# shared log files
LOGS = dict()