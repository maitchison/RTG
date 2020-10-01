"""
Hard coded stratigies for rescue the general.

Note: some of these stratigies are allowed to cheat (i.e have access to global or hidden information)

"""

from rescue import RescueTheGeneralEnv

class RTG_ScriptedEnv(RescueTheGeneralEnv):
    """
    Allows some players within the RTG environment to be scripted
    """

    def __init__(self, scenario="full"):
        super().__init__(scenario)

    def step(self, actions):
        pass

    def reset(self):

        return super().reset()



def seek_and_destroy(player_vision, env):
    """
    Roam around map looking for enemy players to destroy
    :return:
    """

    # randomly pick a spot and navigate to it

    # if enemy target is vision path to them

    # if enemy is shootable shoot them

    # note: how to deal with gray targets

    pass


def random(player_vision, env):
    """
    Agent presses random keys
    :return: the action
    """
    pass

def wander(player_vision, env):
    """
    Agent wanders around map
    :return: the action
    """
    pass

def follow(player_vision, env):
    """
    Agent tries to follow other agents
    :return: the action
    """
    pass

def save_general(player_vision, env):
    """
    Agent rushes to general to save them
    :return: the action
    """
    pass

def rush_general(player_vision, env):
    """
    Agent rushes to general to hill them
    :return: the action
    """
    pass


def avoid(player_vision, env):
    """
    Agent attempts to avoid other soldiers
    :return: the action
    """
    pass
