"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

import os

from stable_baselines.common import ActorCriticRLModel

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # errors only
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from rescue import RescueTheGeneralEnv
from MARL import MultiAgentVecEnv
from tools import load_results, get_score, get_score_alt, export_graph

from typing import Union, List
import json
import pickle
import uuid
import numpy as np
import cv2
import os
import argparse
import time
import shutil
from ast import literal_eval
import gc

import strategies
from strategies import RTG_ScriptedEnv

from stable_baselines.common.policies import CnnLstmPolicy as POLICY
from stable_baselines import PPO2 as PPO

#from ppo_ma import PPO_MA as PPO
#from policy_ma import CnnLstmPolicy_MA as POLICY


import rescue
from new_models import cnn_default, cnn_fast

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

class Config():
    """ Class to hold config files"""

    default_algo_params = {
        'learning_rate': 2.5e-4,
        'n_steps': 128,     # this does seem to help, 40 is fine sometimes...
        'ent_coef': 0.01,
        'n_cpu_tf_sess': 1,
        'nminibatches': 8,  # used to be 4 but needs to be 8 so we can fit into memory with larger n_steps
        'max_grad_norm': 5.0,
        'gamma': 0.995, # 0.998 was used in openAI hide and seek paper
        'cliprange_vf': -1  # this has been shown to not be effective so I disable it
    }

    def __init__(self):
        self.log_folder = str()
        self.device = str()
        self.epochs = int()
        self.model = str()
        self.cpus = 1 if tf.test.is_gpu_available else None  # for GPU machines its best to use only 1 CPU core
        self.parallel_envs = int()
        self.algo_params = dict()
        self.run = str()
        self.force_cpu = bool()
        self.script_blue_team = str()
        self.export_video = bool()
        self.train_scenarios = list()
        self.eval_scenarios = list()
        self.vary_team_player_counts = bool()

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

    def setup(self, args):

        config_vars = set(k for k,v in vars(self).items())

        # setup config from command line args
        # most of these just get copied across directly
        for arg_k,arg_v in vars(args).items():
            # check if this matches a config variable
            if arg_k in config_vars:
                vars(self)[arg_k] = arg_v

        self.uuid = uuid.uuid4().hex[-8:]
        if args.mode == "evaluate":
            self.log_folder = f"run/{args.run}/evaluate [{self.uuid}]"
        else:
            self.log_folder = f"run/{args.run} [{self.uuid}]"
        rescue.LOG_FILENAME = self.log_folder
        self.algo_params = self.default_algo_params.copy()
        self.algo_params.update(literal_eval(args.algo_params))

        # setup the scenarios... these are a bit complex now due to the scripted players
        args.eval_scenarios = args.eval_scenarios or args.train_scenarios
        config.train_scenarios = ScenarioSetting.parse(args.train_scenarios)
        config.eval_scenarios = ScenarioSetting.parse(args.eval_scenarios)

        if self.device.lower() != "auto":
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device

        # Set CPU as available physical device
        if config.force_cpu:
            print("Forcing CPU")
            my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
            tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')


def flexiable_step(model, env_states, agent_states, env_dones):
    """
    Stable-baselines has a limitation that models must always have the same batch size.
    Sometimes we want to run the model with more or less agents. This function allows for that.

    This would be much better if the model simply supported a flexiable input size.

    :param env_states:
    :param agent_states:
    :param env_dones:
    :return:
    """

    required_n = len(env_states)

    # note, this requirement can be relaxed in the future by running the model multiple times
    assert required_n <= model.n_envs, f"Tried to process {required_n} agents but model supports a maximum of {model.n_envs}"

    env_state_shape = env_states[0].shape
    agent_state_shape = agent_states[0].shape

    padded_states = np.zeros((model.n_envs, *env_state_shape), dtype=env_states.dtype)
    padded_agent_states =  np.zeros((model.n_envs, *agent_state_shape), dtype=agent_states.dtype)
    padded_dones = np.zeros((model.n_envs,), dtype=np.bool)

    padded_states[:required_n] = env_states
    padded_agent_states[:required_n] = agent_states
    padded_dones[:required_n] = env_dones

    actions, _, agent_states, _ = model.step(padded_states, padded_agent_states, padded_dones)

    return actions[:required_n], agent_states[:required_n]


def evaluate_model(model:ActorCriticRLModel, eval_scenario, sub_folder, trials=100):
    """
    Evaluate given model in given environment.
    :param model:
    :param vec_env:
    :param trials:
    :return:
    """

    # run them all in parallel at once to make sure we get exactly 'trials' number of environments
    vec_env = make_env(eval_scenario, name="eval", log_path=sub_folder, vary_players=False, parallel_envs=trials)
    env_states = vec_env.reset()
    agent_states = model.initial_state[:vec_env.num_envs]
    env_dones = np.zeros([len(agent_states)], dtype=np.bool)
    vec_env.run_once = True

    # play the game...
    results = [[0,0,0] for _ in range(trials)]
    while not all(env_dones):
        actions, agent_states = flexiable_step(model, env_states, agent_states, env_dones)
        env_states, env_rewards, env_dones, env_infos = vec_env.step(actions)

        # look for finished games
        for i, env in enumerate(vec_env.envs):
            if env.outcome != "":
                results[i] = env.team_scores

    # collate results
    red_score = np.mean([r for r, g, b in results])
    green_score = np.mean([g for r, g, b in results])
    blue_score = np.mean([b for r, g, b in results])

    return red_score, green_score, blue_score


def export_video(filename, model, scenario):
    """
    Exports a movie of model playing in given scenario
    """

    scale = 8

    # make a new environment so we don't mess the settings up on the one used for training.
    # it also makes sure that results from the video are not included in the log
    vec_env = make_env(scenario, parallel_envs=1, name="video")

    env_states = vec_env.reset()
    env = vec_env.envs[0]
    frame = env.render("rgb_array")

    # work out our height
    height, width, channels = frame.shape
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height), isColor=True)

    # get initial states
    agent_states = model.initial_state

    # this is required to make sure the last frame is visible
    vec_env.auto_reset = False

    env_state_shape = env_states.shape[1:]
    states = np.zeros((model.n_envs, *env_state_shape), dtype=np.uint8)

    # play the game...
    while env.outcome == "":

        # model expects parallel_agents environments but we're running only 1, so duplicate...
        # (this is a real shame, would be better if this just worked with a flexable input size)

        # state will be [n, h, w, c], this process will duplicate it.
        states[:env.n_players] = env_states
        actions, _, agent_states, _ = model.step(states, agent_states, [False] * model.n_envs)

        env_states, env_rewards, env_dones, env_infos = vec_env.step(actions[:env.n_players])

        # generate frames from global perspective
        frame = env.render("rgb_array")

        # for some reason cv2 wants BGR instead of RGB
        frame[:, :, :] = frame[:, :, ::-1]

        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert \
            frame.shape[1] == width and frame.shape[0] == height, \
            "Frame should be {} but is {}".format((width, height, 3), frame.shape)

        video_out.write(frame)

    video_out.release()

    # rename video to include outcome
    try:
        modified_filename = f"{os.path.splitext(filename)[0]} [{env.outcome}]{os.path.splitext(filename)[1]}"
        shutil.move(filename, modified_filename)
    except:
        print("Warning: could not rename video file.")


def make_env(scenarios: Union[List[ScenarioSetting], ScenarioSetting, str], parallel_envs = None, log_path = None,
             name="env", vary_players = False):
    """
    Creates a vectorized environment from given scenario specifications
    :param scenarios: Either a string: e.g. "red2", in which case a single scenario with no scripting is used, or a
        single ScriptedScenario, or a list of ScriptedScenarios
    :param parallel_envs: Number of times to duplicate the environment(s)
    :param enable_logging:
    :param name:
    :return:
    """

    # for convenience we allow non-list input, and string format
    if type(scenarios) is ScenarioSetting:
        scenarios = [scenarios]

    if type(scenarios) is str:
        scenarios = ScenarioSetting.parse(scenarios)

    parallel_envs = parallel_envs or config.parallel_envs

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
                    log_file=_log_file,
                    dummy_prob=0.25 if vary_players else 0
                )

            env_functions.append(make_env_fn)

    vec_env = MultiAgentVecEnv(env_functions)

    return vec_env

def get_current_epoch():
    return CURRENT_EPOCH

def train_model():
    """
    Train model on the environment using the "other agents are environment" method.
    :return:
    """

    print("="*60)

    # copy source files for later
    from shutil import copyfile
    for filename in ["train.py", "rescue.py"]:
        copyfile(filename, f"{config.log_folder}/{filename}")

    # make a copy of the environment parameters
    with open(f"{config.log_folder}/config.txt", "w") as f:
        f.write(str(config))

    vec_env = make_env(config.train_scenarios, name="train", log_path=config.log_folder,
                       vary_players=config.vary_team_player_counts)

    print("Scenario parameters:")
    scenario_descriptions = set(str(env.scenario) for env in vec_env.envs)
    for description in scenario_descriptions:
        print(description)
    print()
    print("Config:")
    print(config)
    print()

    model = make_model(vec_env)

    print("="*60)

    start_time = time.time()

    step_counter = 0

    for epoch in range(0, config.epochs):

        global CURRENT_EPOCH
        CURRENT_EPOCH = epoch

        print()
        print(f"Training epoch {epoch} on experiment {config.log_folder}")

        # perform evaluations (if required)
        for index, eval_scenario in enumerate(config.eval_scenarios):


            sub_folder = f"{config.log_folder}/eval_{index}"
            os.makedirs(sub_folder, exist_ok=True)
            results_file = os.path.join(sub_folder, "results.csv")

            scores = evaluate_model(model, eval_scenario, sub_folder, trials=100)
            rounded_scores = tuple(round(float(score), 1) for score in scores)

            print(f" -evaluation against {str(eval_scenario):<40} {rounded_scores}")

            # generate a video
            if config.export_video:
                export_video(f"{sub_folder}/evaluation_{epoch:03}_M.mp4", model, eval_scenario)

            # write results to text file
            if not os.path.exists(results_file):
                with open(results_file, "w") as f:
                    f.write("epoch, red_score, green_score, blue_score\n")
            with open(results_file, "a+") as f:
                f.write(f"{epoch}, {scores[0]}, {scores[1]}, {scores[2]}\n")

            # flush buffer
            rescue.flush_logs()

            try:
                log_file = os.path.join(sub_folder, f"env_0.csv")
                export_graph(log_file, epoch=epoch, png_base_name=f"eval_{index}")
            except Exception as e:
                # not worried about this not working...
                print(e)

        # export training video
        if config.export_video:
            export_video(f"{config.log_folder}/training_{epoch:03}_M.mp4", model, config.train_scenarios[0])
        model.save(f"{config.log_folder}/model_{epoch:03}_M.p")

        sub_epoch = 0

        step_counter = learn(model, step_counter, (epoch+1)*1e6)
        print()

        # flush the log buffer and print scores
        rescue.flush_logs()
        print_scores(epoch=epoch)

    model.save(f"{config.log_folder}/model_final.p")
    if config.export_video:
        export_video(f"{config.log_folder}/ppo_run_{config.epochs:03}_M.mp4", model, config.train_scenarios[0])

    time_taken = time.time() - start_time
    print(f"Finished training after {time_taken/60/60:.1f}h.")

def make_model(vec_env: MultiAgentVecEnv, model_name = None, verbose=0):

    model_name = model_name or config.model

    # figure out a mini_batch size
    # batch_size = config.algo_params["n_steps"] * vec_env.num_envs
    # assert batch_size % config.algo_params["mini_batch_size"] == 0, \
    #     f"Invalid mini_batch size {config.algo_params['mini_batch_size']}, must divide batch size {batch_size}."
    # n_mini_batches = batch_size // config.algo_params["mini_batch_size"]
    # print(f" -using {n_mini_batches} mini-batch(s) of size {config.algo_params['mini_batch_size']}.")

    batch_size = config.algo_params["n_steps"] * vec_env.num_envs
    mini_batch_size = batch_size // config.algo_params["nminibatches"]
    print(f"Model created, using batch size of {batch_size} and mini-batch size of {mini_batch_size}")

    params = config.algo_params.copy()

    if model_name == "cnn_lstm_default":
        policy_kwargs = {
            "cnn_extractor": cnn_default,
            "n_lstm": 128
        }

    elif model_name == "cnn_lstm_fast":
        policy_kwargs = {
            "cnn_extractor": cnn_fast,
            "n_lstm": 64
        }
    else:
        raise ValueError(f"Invalid model name {model_name}")

    params["verbose"] = verbose
    params["policy_kwargs"] = policy_kwargs

    model = PPO(POLICY, vec_env, **params)

    return model

def run_benchmarks(train=True, model=True, env=True):

    def bench_scenario(scenario_name):

        vec_env = make_env(scenario_name, name="benchmark")

        _ = vec_env.reset()
        steps = 0

        start_time = time.time()

        while time.time() - start_time < 10:
            random_actions = np.random.randint(0, 10, size=[vec_env.num_envs])
            states, _, _, _ = vec_env.step(random_actions)
            steps += vec_env.num_envs

        time_taken = (time.time() - start_time)

        print(f" - scenario {scenario_name} runs at {steps / time_taken / 1000:.1f}k FPS.")

    def bench_training(scenario_name, model_name):

        vec_env = make_env(scenario_name, name="benchmark")
        model = make_model(vec_env, model_name, verbose=0)

        # just to warm it up
        model.learn(model.n_batch)

        start_time = time.time()
        model.learn(2 * model.n_batch)
        time_taken = (time.time() - start_time)

        print(f" - model {model_name} trains at {2 * model.n_batch / time_taken / 1000:.1f}k FPS.")

    def bench_model(model_name):

        vec_env = make_env("full")
        model = make_model(vec_env, model_name)

        states = np.asarray(vec_env.reset())

        model_states = model.initial_state
        model_masks = np.zeros((vec_env.num_envs,), dtype=np.uint8)
        steps = 0

        start_time = time.time()

        while time.time() - start_time < 10:

            actions, _, model_states, _ = model.step(states, model_states, model_masks)
            steps += vec_env.num_envs

        time_taken = (time.time() - start_time)

        print(f" - model {model_name} runs at {steps / time_taken / 1000:.1f}k FPS.")

    if train:
        print("Benchmarking training...")
        for model_name in ["cnn_lstm_default", "cnn_lstm_fast"]:
            bench_training("red2", model_name)

    if env:
        print("Benchmarking environments...")
        for scenario_name in ["full", "medium", "red2"]:
            bench_scenario(scenario_name)

    if model:
        print("Benchmarking models (inference)...")
        for model_name in ["cnn_lstm_default", "cnn_lstm_fast"]:
            bench_model(model_name)

def print_scores(epoch=None):
    """ Print current scores, also makes a plot"""
    try:
        log_file = os.path.join(config.log_folder, "env_0.csv")
        results = load_results(log_file)
        scores = tuple(round(get_score(results, team), 1) for team in ["red", "green", "blue"])
        print(f" -training scores: {scores}")
        export_graph(log_file, epoch=epoch, png_base_name="train")
    except:
        # this usually just means results have not generated yet
        pass

def load_model(filename, env=None):
    """
    Loads model from given epoch checkpoint
    Loads model from given epoch checkpoint
    :param filename:
    :return:
    """
    model = PPO.load(filename, env)
    return model

def learn(model, step_counter, max_steps, verbose=True):

    sub_epoch = 0

    while step_counter < max_steps:

        global CURRENT_EPOCH
        CURRENT_EPOCH = step_counter / 1e6

        # silly baselines, will round down to nearest batch. Our batches are often around 49k, so running
        # .learn(100000) will only actually generate 88k steps. To keep all the numbers correct I therefore
        # run each batch individually. For large batch sizes this should be fine.
        learn_steps = model.n_batch
        start_epoch_time = time.time()
        model.learn(learn_steps, reset_num_timesteps=step_counter == 0)
        step_counter += learn_steps
        epoch_time = time.time() - start_epoch_time

        fps = learn_steps / epoch_time

        # ignore first sub-epoch as FPS will be lower than normal
        if verbose:
            if sub_epoch == 1:
                print(f" -FPS: {fps:.0f} .", end='', flush=True)
            elif sub_epoch > 1:
                print(".", end='', flush=True)

        # this is needed to stop a memory leak
        gc.collect()

        sub_epoch += 1

    return step_counter


def regression_test():

    print("Performing regression test, this could take some time.")

    start_time = time.time()

    def run_test(scenario_name, n_steps=int(2e6)):

        destination_folder = os.path.join(config.log_folder, scenario_name)
        os.makedirs(destination_folder, exist_ok=True)
        log_file = os.path.join(destination_folder, "env_0.csv")

        # our MARL environments are handled like vectorized environments
        make_env = lambda: RescueTheGeneralEnv(scenario_name=scenario_name, name="test", log_file=log_file)
        vec_env = MultiAgentVecEnv([make_env for _ in range(config.parallel_envs)])

        model = make_model(vec_env, verbose=0)

        learn(model, 0, n_steps)
        print()

        export_video(f"{destination_folder}/{scenario_name}.mp4", model, scenario_name)

        # flush the log buffer
        rescue.flush_logs()

        try:
            export_graph(log_file, epoch=2, png_base_name="results")
        except Exception as e:
            # not worried about this not working...
            print(e)

        # return scores
        return load_results(log_file)


    for scenario_name, team, required_score in [
        ("red2", "red", 7.5),
        ("green2", "green", 7.5),
        ("blue2", "blue", 7.5),
    ]:

        results = run_test(scenario_name)
        score = get_score(results, team)
        score_alt = get_score_alt(results, team)

        result = "FAIL" if score < required_score else "PASS"
        print(f"  [{result:}] {scenario_name:<20} ({score:.1f}, {score_alt:.1f})")

    time_taken = time.time() - start_time
    print(f"Finished tests in {time_taken/60:.1f}m.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="[bench|train|test|evaluate]")
    parser.add_argument('--run', type=str, help="run folder", default="test")
    parser.add_argument('--device', type=str, help="[0|1|2|3|AUTO]", default="auto")

    parser.add_argument('--train_scenarios', type=str, default="full",
        help="Scenario settings for training. Can be a single scenario name e.g. 'red2' or for a mixed training setting "
             +" use [['<scenario>', '<red_strat>', '<green_strat>', '<blue_strat>'], ...]")
    parser.add_argument('--eval_scenarios', type=str,
        help="Scenario settings used to evaluate (defaults to same as train_scenario)", default=None)

    parser.add_argument('--epochs', type=int, help="number of epochs to train for (each 1M agent steps)", default=500)
    parser.add_argument('--model', type=str, help="model to use [cnn_lstm_default|cnn_lstm_fast]", default="cnn_lstm_default")
    parser.add_argument('--force_cpu', type=bool, default=False)
    parser.add_argument('--script_blue_team', type=str, default=None)
    parser.add_argument('--export_video', type=bool, default=True)
    parser.add_argument('--algo_params', type=str, default="{}")
    parser.add_argument('--parallel_envs', type=int, default=128,
                        help="The number of times to duplicate the environments. Note: when using mixed learning each"+
                             "environment will be duplicated this number of times.")
    parser.add_argument('--vary_team_player_counts', type=bool, default=False, help="Use a random number of players turning training.")

    args = parser.parse_args()
    config.setup(args)

    print()
    print(f"Starting {config.log_folder} on device {config.device}")

    os.makedirs(config.log_folder, exist_ok=True)

    if args.mode == "bench":
        run_benchmarks()
    elif args.mode == "bench_env":
        run_benchmarks(env=True, model=False, train=False)
    elif args.mode == "bench_model":
        run_benchmarks(env=False, model=True, train=False)
    elif args.mode == "bench_train":
        run_benchmarks(env=False, model=False, train=True)
    elif args.mode == "train":
        train_model()
    elif args.mode == "test":
        regression_test()
    else:
        raise Exception(f"Invalid mode {args.mode}")


if __name__ == "__main__":
    CURRENT_EPOCH = 0
    RescueTheGeneralEnv.get_current_epoch = get_current_epoch
    config = Config()
    main()