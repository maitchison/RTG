import os
import uuid
import torch
import numpy as np
import sys

from typing import Union, List
from ast import literal_eval
from scenarios import ScenarioSetting, RescueTheGeneralScenario
from strategies import RTG_ScriptedEnv
from algorithms import MarlAlgorithm, PMAlgorithm, MultiAgentVecEnv

import strategies
import rescue
import utils

class Config():
    """ Class to hold config files"""

    def __init__(self):
        self.log_folder = str()
        self.device = str()
        self.epochs = int()
        self.model = str()
        self.parallel_envs = int()
        self.algo_params = dict()
        self.run = str()
        self.force_cpu = bool()
        self.script_blue_team = str()
        self.export_video = bool()
        self.train_scenarios = list()
        self.eval_scenarios = list()
        self.amp = bool()
        self.micro_batch_size: Union[str, int] = str()
        self.n_steps = int()
        self.export_rollout = bool()
        self.test_epochs = int()
        self.save_model = str()
        self.prediction_mode = str()
        self.deception_bonus = tuple()
        self.split_policy = bool()
        self.nan_check = bool()
        self.use_global_value = bool()
        self.seed = int()
        self.restore = bool()
        self.cmd_args = str()

        self.verbose = int()

    def __str__(self):

        # custom one looks better and will evaluate ok using literal_eval
        lines = []
        for k,v in vars(self).items():
            key_string = f"'{k}':"
            if type(v) is str: # wrap strings in quotes
                v = f"'{v}'"
            lines.append(f"{key_string:<20}{v},")
        return "{\n"+("\n".join(lines))+"\n}"

        # d = {}
        # for k,v in vars(self).items():
        #     if type(v) is list and len(v) > 0 and type(v[0]) is ScenarioSetting:
        #         v = [[scenario.scenario_name] + scenario.strategies for scenario in v]
        #     d[k] = v
        # return json.dumps(d, indent=4)

    def setup(self, args:dict):

        self.cmd_args = " ".join(sys.argv)

        config_vars = set(k for k,v in vars(self).items())

        if type(args['algo_params']) is str:
            args['algo_params'] = literal_eval(args['algo_params'])

        # setup config from command line args
        # most of these just get copied across directly
        for arg_k,arg_v in args.items():
            # check if this matches a config variable
            if arg_k in config_vars:
                if type(arg_v) is str:
                    # map all strings to lower_case
                    arg_v = arg_v.lower()
                vars(self)[arg_k] = arg_v

        self.uuid = uuid.uuid4().hex[-8:]

        if 'log_folder' not in args:
            if args.get('mode','') == "evaluate":
                self.log_folder = f"run/{args['run']}/evaluate [{self.uuid}]"
            else:
                self.log_folder = f"run/{args['run']} [{self.uuid}]"

        rescue.LOG_FILENAME = self.log_folder

        # work out the device
        if self.device == "auto":
            self.device = "cuda" if torch.has_cuda else "cpu"

        if type(self.deception_bonus) == str:
            self.deception_bonus = literal_eval(str(self.deception_bonus))
            if type(self.deception_bonus) in [list]:
                self.deception_bonus = tuple(self.deception_bonus)

        if type(self.deception_bonus) in [float, int]:
            self.deception_bonus = tuple([self.deception_bonus] * 3)

        # setup the scenarios... these are a bit complex now due to the scripted players
        args['eval_scenarios'] = args.get('eval_scenarios', args.get('train_scenarios'))
        self.train_scenarios = ScenarioSetting.parse(args['train_scenarios'])
        self.eval_scenarios = ScenarioSetting.parse(args['eval_scenarios'])

        if self.seed < 0:
            self.seed = np.random.randint(0, 99999999)

def make_algo(vec_env: MultiAgentVecEnv, config:Config, model_name=None):

    algo_params = config.algo_params.copy()

    algo_params["model_name"] = model_name or config.model

    algorithm = PMAlgorithm(
        vec_env,
        device=config.device,
        amp=config.amp,
        export_rollout=config.export_rollout,
        micro_batch_size=config.micro_batch_size,
        n_steps=config.n_steps,
        use_global_value_module=config.use_global_value,
        prediction_mode=config.prediction_mode,
        deception_bonus=config.deception_bonus,
        split_policy=config.split_policy,
        verbose=config.verbose >= 2, **algo_params,
        nan_check=config.nan_check,
    )

    algorithm.log_folder = config.log_folder

    print(f" -model created using batch size of {algorithm.batch_size} and mini-batch size of {algorithm.mini_batch_size}")

    return algorithm


def evaluate_model(algorithm: MarlAlgorithm, eval_scenario, sub_folder, trials=100):
    """
    Evaluate given model in given environment.
    :param algorithm:
    :param trials:
    :return:
    """

    # note, this is the evaluation used by train.py, merge this with arean's evaluation script

    # run them all in parallel at once to make sure we get exactly 'trials' number of environments
    os.makedirs(sub_folder, exist_ok=True)
    vec_env = make_env(eval_scenario, trials, name="eval", log_path=sub_folder, )
    env_obs = vec_env.reset()
    rnn_states = algorithm.get_initial_rnn_state(vec_env.num_envs)
    env_terminals = np.zeros([len(rnn_states)], dtype=np.bool)
    vec_env.run_once = True

    # play the game...
    results = [(0, 0, 0) for _ in range(trials)]
    while not all(env_terminals):

        with torch.no_grad():
            roles = vec_env.get_roles()
            model_output, new_rnn_states = algorithm.forward(
                obs=torch.from_numpy(env_obs),
                rnn_states=rnn_states,
                roles=torch.from_numpy(roles)
            )
            rnn_states[:] = new_rnn_states

            log_policy = model_output["log_policy"].detach().cpu().numpy()
            actions = utils.sample_action_from_logp(log_policy)

        env_obs, env_rewards, env_terminals, env_infos = vec_env.step(actions)

        # look for finished games
        for i, env in enumerate(vec_env.games):
            if env.round_outcome != "":
                results[i] = env.round_team_scores

    # collate results
    red_score = np.mean([r for r, g, b in results])
    green_score = np.mean([g for r, g, b in results])
    blue_score = np.mean([b for r, g, b in results])

    # make sure results have be written to env log
    rescue.flush_logs()

    return red_score, green_score, blue_score


def make_env(
        scenarios: Union[List[ScenarioSetting], ScenarioSetting, str],
        parallel_envs:int,
        log_path=None,
        name="env"
):
    """
    Creates a vectorized environment from given scenario specifications
    :param scenarios: Either a string: e.g. "red2", in which case a single scenario with no scripting is used, or a
        single ScriptedScenario, or a list of ScriptedScenarios
    :param parallel_envs: Number of times to duplicate the environment(s)
    :param name:
    :return:
    """

    # for convenience we allow non-list input, and string format
    if isinstance(scenarios, ScenarioSetting):
        scenarios = [scenarios]

    if isinstance(scenarios, str):
        scenarios = ScenarioSetting.parse(scenarios)

    env_functions = []
    for _ in range(parallel_envs):
        for index, scenario_setting in enumerate(scenarios):
            # convert strategies names to strategy functions
            strats = []
            for strategy in scenario_setting.strategies:
                if strategy is not None:
                    strats.append(strategies.register[strategy])
                else:
                    strats.append(None)

            if log_path is None:
                log_file = None
            else:
                log_file = os.path.join(log_path, f"env_{index}.csv")

            make_env_fn = lambda _strats=tuple(strats), _name=name, _scenario_setting=scenario_setting, _log_file=log_file: \
                RTG_ScriptedEnv(
                    scenario_name=_scenario_setting.scenario_name, name=_name,
                    red_strategy=_strats[0],
                    green_strategy=_strats[1],
                    blue_strategy=_strats[2],
                    log_file=_log_file
                )

            env_functions.append(make_env_fn)

    vec_env = MultiAgentVecEnv(env_functions)

    return vec_env
