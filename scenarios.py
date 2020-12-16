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
        self.timeout_rnd = 0.1      # this helps make sure games are not always in sync, which can happen if lots of
                                    # games timeout.
        self.general_always_visible = False
        self.general_initial_health = 10
        self.player_initial_health = 10
        self.location_encoding = "abs"  # none | sin | abs
        self.battle_royale = False   # removes general from game, and adds kill rewards
        self.bonus_actions = False   # provides small reward for taking an action that is indicated on agents local
                                     # observation some time after the signal appeared
        self.bonus_actions_one_at_a_time = False
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

