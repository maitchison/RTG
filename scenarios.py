from ast import literal_eval
import numpy as np

class ScenarioSetting():
    """
    Scenario paired with optional scripted behaviour for teams
    """
    def __init__(self, scenario_name, strategies):
        self.scenario_name = scenario_name
        self.strategies = strategies

    @staticmethod
    def parse(input):
        """
        Converts text into an array of scenarios
        :param input:
        :return:
        """
        if '[' not in input:
            input = f"[['{input}', None, None, None]]"

        input = literal_eval(input)

        result = []

        for scenario_info in input:
            scenario = ScenarioSetting(scenario_info[0], scenario_info[1:4])
            result.append(scenario)

        return result

    def __repr__(self):
        array = [self.scenario_name, *self.strategies]
        return str(array)


class RescueTheGeneralScenario():

    SCENARIOS = {

        "alt": {
            "description": "Alternative version of the game",
            "map_width": 48,
            "map_height": 48,
            "team_counts": (1, 2, 3),
            "n_trees": 10,
            "reward_per_tree": 1,
            "hidden_roles": "default",
            "timeout_mean": 1000,             # this gives red enough time to find the general, otherwise blue might learn to force a draw.
            "max_view_distance": 6,
            "team_general_view_distance": (2, 5, 5),  # how close you need to be to the general to see them
            "team_shoot_damage": (2, 2, 10),  # blue can 1-shot other players, but red and green can not.
            "team_view_distance": (6, 5, 5),  # red can see further
            "team_shoot_range": (5, 5, 5),
            "general_initial_health": 9999,  # general can not be killed
            "players_to_move_general": 2,
            "blue_general_indicator": "direction",
            "starting_locations": "together",  # players start together
            "team_shoot_timeout": (2, 2, 2),
            "red_wins_if_sees_general": True,   # blue must make sure red never reveals the location of the general
        },

        "full": {
            "description": "The full game",
            "map_width": 48,
            "map_height": 48,
            "team_counts": (2, 2, 2),
            "n_trees": 10,
            "reward_per_tree": 1,
            "hidden_roles": "default",
            "timeout_mean": 500,
            "max_view_distance": 6,
            "team_shoot_damage": (10, 5, 5),
            "team_view_distance": (6, 5, 5),    # red can see further
            "team_shoot_range": (5, 5, 5),
            "starting_locations": "together",   # players start together
            "team_shoot_timeout": (20, 20, 20),
        },

        "blue4": {
            "description": "four blue two red, no green, smaller map",
            "map_width": 32,
            "map_height": 32,
            "team_counts": (2, 0, 4),
        },

        "r2g2": {
            "description": "Two red players and two green players on a small map",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 2, 0),
            "n_trees": 10,
            "reward_per_tree": 1,
            "hidden_roles": "none",
            "max_view_distance": 5,             # makes thins a bit faster
            "team_view_distance": (5, 5, 5),    # no bonus vision for red
            "team_shoot_damage": (5, 5, 5),     # 2 hits to kill
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",     # random start locations
            "team_shoot_timeout": (5, 5, 5)      # green is much slower at shooting
        },

        "r2g2_rp": {
            "description": "Smaller version of r2g2, used for testing role prediction",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 2, 0),
            "n_trees": 10,
            "reward_per_tree": 1,
            "hidden_roles": "none",
            "max_view_distance": 5,  # makes thins a bit faster
            "team_view_distance": (5, 5, 5),  # no bonus vision for red
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",  # random start locations
            "team_shoot_timeout": (5, 5, 5)  # green is much slower at shooting
        },

        "r2g2_hrp": {
            "description": "Smaller version of r2g2, used for testing role prediction with hidden roles",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 2, 0),
            "n_trees": 10,
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
            "reward_per_tree": 1,
            "hidden_roles": "none",
            "max_view_distance": 5,  # makes thins a bit faster
            "team_view_distance": (5, 5, 5),  # no bonus vision for red
            "team_shoot_range": (4, 4, 4),
            "starting_locations": "random",  # random start locations
            "team_shoot_timeout": (5, 5, 5)  # green is much slower at shooting
        },

        "wolf_sheep": {
            "description": "A wolf among the sheep",
            "map_width": 32,
            "map_height": 32,
            "team_counts": (1, 3, 0),
            "n_trees": 18,
            "reward_per_tree": 0.5,
            "hidden_roles": "all",
            "timeout_mean": 400,               # make sure games don't last too long, 400 is plenty of time for green
                                               # to harvest all the trees
            "max_view_distance": 5,            # makes things a bit faster having smaller vision
            "team_view_distance": (5, 5, 5),   # no bonus vision for red
            "team_shoot_range": (3, 0, 0),
            "starting_locations": "together",  # random start locations
            "team_shoot_timeout": (20, 0, 0),
            "enable_voting": True,
            #"frame_blanking": 0.75,             # stub: delete 75% of frames
            "battle_royale": True,              # removes general, and ends game if all green players are killed, or
                                                # if green eliminates red players and harvests all trees
            "voting_button": True,              # place a voting button at the start
            "zero_sum": True,
            "points_for_kill": np.asarray((     # loose a point for self kill, gain one for other team kill
                (-1, +3.33, +1),
                (+1, -1, +1),
                (+1, +1, -1),
            ))
        },

        "5p": {
            "description": "A five player game",
            "map_width": 48,
            "map_height": 48,
            "team_counts": (1, 2, 2),
            "n_trees": 18,
            "reward_per_tree": 0.5,
            "hidden_roles": "all",
            "timeout_mean": 400,  # make sure games don't last too long, 400 is plenty of time for green
            # to harvest all the trees
            "max_view_distance": 5,  # makes things a bit faster having smaller vision
            "team_shoot_damage": (10, 5, 5),
            "team_view_distance": (5, 5, 5),  # no bonus vision for red
            "team_shoot_range": (5, 5, 5),
            "starting_locations": "together",  # random start locations
            "team_shoot_timeout": (20, 20, 20),
            "enable_voting": False,
            "voting_button": False,          # place a voting button at the start
        },

        "wolf_sheep_nv": {
            "description": "A wolf among the sheep, no voting",
            "map_width": 32,
            "map_height": 32,
            "team_counts": (1, 3, 0),
            "n_trees": 18,
            "reward_per_tree": 0.5,
            "hidden_roles": "all",
            "timeout_mean": 400,  # make sure games don't last too long, 400 is plenty of time for green
            # to harvest all the trees
            "max_view_distance": 5,  # makes things a bit faster having smaller vision
            "team_view_distance": (5, 5, 5),
            "team_shoot_damage": (10, 5, 5),
            "team_shoot_range": (5, 5, 5),
            "starting_locations": "together",  # random start locations
            "team_shoot_timeout": (20, 20, 20),
            "enable_voting": False,
            "zero_sum": True,
            "battle_royale": True,  # removes general, and ends game if all green players are killed, or
            # if green eliminates red players and harvests all trees
            "points_for_kill": np.asarray((  # loose a point for self kill, gain one for other team kill
                (-1, +3.33, +1),
                (+1, -1, +1),
                (+1, +1, -1),
            ))
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

        "mem2": {
            "description": "A test to make sure memory works.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "bonus_actions": True,
            "bonus_actions_delay": 2,
            "timeout_mean": 200,
            "player_initial_health": 9999,
            "general_initial_health": 9999,  # game won't end until timeout
        },

        "mem2b": {
            "description": "A test to make sure memory works.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "bonus_actions": True,
            "bonus_actions_one_at_a_time": True,
            "bonus_actions_delay": 2,
            "timeout_mean": 200,
            "player_initial_health": 9999,
            "general_initial_health": 9999,  # game won't end until timeout
        },

        "mem4": {
            "description": "A test to make sure memory works.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "bonus_actions": True,
            "bonus_actions_delay": 4,
            "timeout_mean": 200,
            "player_initial_health": 9999,
            "general_initial_health": 9999,  # game won't end until timeout
        },

        "mem4b": {
            "description": "A test to make sure memory works.",
            "map_width": 24,
            "map_height": 24,
            "team_counts": (2, 0, 0),
            "max_view_distance": 5,
            "team_view_distance": (5, 5, 5),
            "bonus_actions": True,
            "bonus_actions_one_at_a_time": True,
            "bonus_actions_delay": 4,
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
            "points_for_kill": np.asarray((  # loose a point for self kill, gain one for other team kill
                (-1, +1, +1),
                (+1, -1, +1),
                (+1, +1, -1),
            )),
            "reveal_team_on_death": True
        }
    }

    # add training verisons
    for k,v in SCENARIOS.copy().items():
        training_key = k+"_training"
        if training_key not in SCENARIOS:
            SCENARIOS[training_key] = {**v, **{"initial_random_kills": 0.5}}

    def __init__(self, scenario_name=None, **kwargs):

        # defaults
        self.n_trees = 20
        self.reward_per_tree = 0.5
        self.map_width = 48
        self.map_height = 48

        self.max_view_distance = 7      # distance used for size of observational space, unused tiles are blanked out
        self.team_view_distance = (7, 5, 5)
        self.team_shoot_damage = (10, 10, 10)
        self.team_general_view_distance = (3, 5, 5) # how close you need to be to the general to see him
        self.team_shoot_range = (4, 0, 0)
        self.team_counts = (4, 4, 4)
        self.team_shoot_timeout = (3, 3, 3)  # number of turns between shooting
        self.enable_voting = False      # enables the voting system
        self.auto_shooting = False    # shooting auto targets closest player
        self.zero_sum = False           # if enabled any points scored by one team will be counted as negative points for all other teams.

        self.timeout_mean = 500
        self.timeout_rnd = 0        # this helps make sure games are not always in sync, which can happen if lots of
                                    # games timeout.
        self.general_initial_health = 10
        self.player_initial_health = 10
        self.battle_royale = False   # removes general from game, and teams can win by eliminating all others teams
        self.bonus_actions = False   # provides small reward for taking an action that is indicated on agents local
                                     # observation some time after the signal appeared
        self.bonus_actions_one_at_a_time = False
        self.bonus_actions_delay = 10
        self.enable_signals = False
        self.starting_locations = "together"
        self.voting_button = False  # creates a voting button near start
        # enables team colors on agents local observation. This can be useful if one policy plays all 3 teams,
        # however it could cause problems if you want to infer what a different team would have done in that situation
        self.local_team_colors = True
        self.frame_blanking = 0     # fraction of frames to zero out (tests memory)
        self.initial_random_kills = 0 # enables random killing of players at the start of the game, can be helpful to make winning viable for a team.

        self.players_to_move_general = 1 # number of players required to move the general
        self.red_wins_if_sees_general = False

        # how many point a player gets for killing a player
        # ordered as points_for_kill[shooting_player_team, killed_player_team]
        self.points_for_kill = np.asarray((
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0)
        ))

        # number of times to soft reset game before a hard reset
        # during a soft reset, player positions and health are reset, but not their teams, or id_colors
        # this allows players to remember roles across soft resets
        # a done is issued only at the end of all resets
        self.rounds = 1

        # default is red knows red, but all others are hidden
        # all is all roles are hidden
        # none is all roles are visible

        self.hidden_roles = "default"

        self.blue_general_indicator = "direction"

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

