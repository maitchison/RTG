"""
Hard coded stratigies for rescue the general.

Note: these stratagies control hte player directly and do not necessarly follow the normal game observational rules
(i.e. they might know who everyone is.



"""

from gym import Wrapper
from rescue import RescueTheGeneralEnv, RTG_Player
from marl_env import MultiAgentEnv
import rescue as rtg
import numpy as np


class RTG_ScriptedEnv(MultiAgentEnv, Wrapper):
    """
    Wrapper that allows some players within the RTG environment to be scripted.
    Each team gets a separate controller, teams without controllers use actions provided to environment
    """

    def __init__(self, scenario_name="full", name="scripted", red_strategy=None, green_strategy=None, blue_strategy=None,
                 log_file=None, dummy_prob=0, **scenario_kwargs):

        vec_env = RescueTheGeneralEnv(scenario_name=scenario_name, name=name, log_file=log_file, dummy_prob=dummy_prob, **scenario_kwargs)
        super(MultiAgentEnv, self).__init__(vec_env)

        self.controllers = [red_strategy, green_strategy, blue_strategy]

        # selects which players observations to pass through
        self.player_filter = []

    @property
    def n_agents(self):
        return sum(count for team, count in enumerate(self.env.scenario.team_counts) if self.controllers[team] is None)

    @property
    def n_players(self):
        return self.env.n_players

    def step(self, actions):

        reversed_actions = actions[::-1]

        all_actions = []

        for player in self.env.players:
            controller = self.controllers[player.team]
            if controller is None:
                action = reversed_actions.pop()
            else:
                action = controller(player, self.env)

            all_actions.append(action)

        obs, rewards, dones, infos = self.env.step(all_actions)

        return obs[self.player_filter],\
            rewards[self.player_filter],\
            dones[self.player_filter],\
            [infos[i] for i in self.player_filter]

    def reset(self):

        obs = self.env.reset()

        # figure out which players to pass through
        # players controlled by a script are excluded
        self.player_filter = []
        for player in self.env.players:
            if self.controllers[player.team] is None:
                self.player_filter.append(player.index)

        return obs[self.player_filter]

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    # pass these through

    @property
    def scenario(self):
        return self.env.scenario

    @property
    def team_scores(self):
        return self.env.round_team_scores

    @property
    def outcome(self):
        return self.env.round_outcome

def stand_still(player:RTG_Player, env:RescueTheGeneralEnv):
    """
    Agent presses random keys
    :return: the action
    """
    return rtg.ACTION_NOOP

def random(player:RTG_Player, env:RescueTheGeneralEnv):
    """
    Agent presses random keys
    :return: the action
    """
    return np.random.choice(range(env.action_space.n))

def wander(player:RTG_Player, env:RescueTheGeneralEnv):
    """
    Agent wanders around map
    :return: the action
    """
    return np.random.choice(list(rtg.MOVE_ACTIONS))

def stand_and_shoot(player:RTG_Player, env:RescueTheGeneralEnv):
    """
    Agent wanders around map
    :return: the action
    """
    return np.random.choice(rtg.SHOOT_ACTIONS)

def save_general(player:RTG_Player, env:RescueTheGeneralEnv):
    """
    Agent rushes to general to save them
    :return: the action
    """
    assert player.team == env.TEAM_BLUE

    # move towards general, but offset a little
    # note right now this will typically only move the general left
    dx = player.x - env.general_location[0]
    dy = player.y - env.general_location[1]

    # we are standing left of general
    if (dx, dy) == (-1, 0):
        return rtg.ACTION_ACT
    else:
        return move_to(player, env.general_location[0]-1, env.general_location[1])

def rush_general_cheat(player:RTG_Player, env:RescueTheGeneralEnv):
    """
    Agent rushes to general to kill them. (not this agent cheats)
    :return: the action
    """
    assert player.team != env.TEAM_BLUE

    shoot_action = fire_at(player, env, *env.general_location)
    if shoot_action != rtg.ACTION_NOOP:
        return shoot_action
    else:
        return move_to(player, *env.general_location)

def rush_general(player:RTG_Player, env:RescueTheGeneralEnv):
    """
    Agent moves randomly until they see the general they they rush / shoot them
    :return: the action
    """
    assert player.team != env.TEAM_BLUE

    dx = player.x - env.general_location[0]
    dy = player.y - env.general_location[1]

    general_in_vision_range = abs(dx) + abs(dy) <= player.view_distance
    general_covered = env.player_at_pos(*env.general_location, include_dead=True) is not None

    if general_in_vision_range and not general_covered:
        return rush_general_cheat(player, env)
    else:
        # pick a random direction to travel in
        if "rush_general_destination" not in player.custom_data:
            player.custom_data["rush_general_destination"] = (
                np.random.randint(0, env.scenario.map_width),
                np.random.randint(0, env.scenario.map_height)
            )
            player.custom_data["rush_general_timer"] = 100

        loc_x, loc_y = player.custom_data["rush_general_destination"]
        action = move_to(player, loc_x, loc_y)
        player.custom_data["rush_general_timer"] -= 1
        # when we reach the spot, or when a timeout occurs, move to next random location
        if action == rtg.ACTION_NOOP or player.custom_data["rush_general_timer"] <= 0:
            del player.custom_data["rush_general_destination"]

        return action

    # -------------------------------------------------
# Helper functions
# -------------------------------------------------

def fire_at(player: RTG_Player, env: RescueTheGeneralEnv, target_x, target_y):
    """
    Returns action to shoot at target, or no-op if target can not be hit.
    :param player:
    :param target_x:
    :param target_y:
    :return:
    """

    dx = player.x - target_x
    dy = player.y - target_y

    if dy == 0:
        if -player.shoot_range <= dx < 0:
            return rtg.ACTION_SHOOT_RIGHT
        elif player.shoot_range >= dx > 0:
            return rtg.ACTION_SHOOT_LEFT

    if dx == 0:
        if -player.shoot_range <= dy < 0:
            return rtg.ACTION_SHOOT_DOWN
        elif player.shoot_range >= dy > 0:
            return rtg.ACTION_SHOOT_UP

    return rtg.ACTION_NOOP


def move_to(player:RTG_Player, target_x, target_y):

    # move towards general
    dx = player.x - target_x
    dy = player.y - target_y

    if dx == dy == 0:
        return rtg.ACTION_NOOP

    # move closer to general
    if abs(dx) >= abs(dy):
        if dx > 0:
            return rtg.ACTION_MOVE_LEFT
        else:
            return rtg.ACTION_MOVE_RIGHT
    else:
        if dy > 0:
            return rtg.ACTION_MOVE_UP
        else:
            return rtg.ACTION_MOVE_DOWN


# -------------------------------------------------
# Not done yet
# -------------------------------------------------

def seek_and_destroy(player, env):
    """
    Roam around map looking for enemy players to destroy
    :return:
    """

    # randomly pick a spot and navigate to it

    # if enemy target is vision path to them

    # if enemy is shootable shoot them

    # note: how to deal with gray targets

    pass

def follow(player, env):
    """
    Agent tries to follow other agents
    :return: the action
    """
    pass


def avoid(player_vision, env):
    """
    Agent attempts to avoid other soldiers
    :return: the action
    """
    pass

register = {
    'wander': wander,
    'random': random,
    'stand_still': stand_still,
    'stand_and_shoot': stand_and_shoot,
    'save_general': save_general,
    'rush_general_cheat': rush_general_cheat,
    'rush_general': rush_general
}