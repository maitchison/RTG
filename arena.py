"""
Fight agents against eachother, and against scripted responses
"""

class Controler():
    """
    Controller for game, plays one side only
    """
    pass

def load_model(path):
    """
    Loads a model from a checkpoint file (.pt) and returns it
    :param path:
    :return:
    """
    pass

def play_game(env: VecEnv, red_controller, green_controller, blue_controller):
    """
    Plays a game with the given controllers. The environment is vectorized so multiple games can be played in parallel.
    Returns the outcome of the game. Optionally a video can be recorded.
    :param env:
    :param red_controller:
    :param green_controller:
    :param blue_controller:
    :return:
    """
    pass

def run_arena(controlers):
    """
    Run an arena with the given controllers.

    Each controller is run against an equal mixture of the other controllers,
    For red and blue their pairwise average scores are recorded, as well as their overall average score.

    :param controlers:
    :return: average score for each controler, and pairwise scores for each red/blue combination
    """

    red_controllers = []
    green_controllers = []
    blue_controllers = []

    env = ...

    for red in red_controllers:
        for blue in blue_controllers:
            for green in green_controllers:
                result[(red, blue)] += play_game(env, red, green, blue)

def main():
    """
    Check agents performance over time
    :return:
    """

    # evaluate against self over time
    for agent in [a1, a2]:
        loop though
        pass

    # evaluate against each other over time (with green being 50/50)

    # evaluate against scripted responses over time

    # run arena at various points in time


