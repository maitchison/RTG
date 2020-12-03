"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

import torch
import torch.cuda
import uuid
import numpy as np
import cv2
import os
import argparse
import time
import shutil
import gc

import strategies
import rescue
import utils

from typing import Union, List
from ast import literal_eval

from rescue import RescueTheGeneralEnv
from marl_env import MultiAgentVecEnv
from tools import load_results, get_score, get_score_alt, export_graph
from strategies import RTG_ScriptedEnv
from algorithms import PMAlgorithm, MarlAlgorithm
from typing import Union
import torch.autograd.profiler as profiler

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
        self.vary_team_player_counts = bool()
        self.amp = bool()
        self.data_parallel = bool()
        self.micro_batch_size: Union[str, int] = str()
        self.n_steps = int()
        self.enable_deception = bool()

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
        self.algo_params = literal_eval(args.algo_params)

        # work out the device
        if config.device.lower() == "auto":
            config.device = "cuda" if torch.has_cuda else "cpu"

        # setup the scenarios... these are a bit complex now due to the scripted players
        args.eval_scenarios = args.eval_scenarios or args.train_scenarios
        config.train_scenarios = ScenarioSetting.parse(args.train_scenarios)
        config.eval_scenarios = ScenarioSetting.parse(args.eval_scenarios)

def evaluate_model(algorithm: MarlAlgorithm, eval_scenario, sub_folder, trials=100):
    """
    Evaluate given model in given environment.
    :param algorithm:
    :param vec_env:
    :param trials:
    :return:
    """

    # run them all in parallel at once to make sure we get exactly 'trials' number of environments
    vec_env = make_env(eval_scenario, name="eval", log_path=sub_folder, vary_players=False, parallel_envs=trials)
    env_obs = vec_env.reset()
    rnn_states = algorithm.get_initial_state(vec_env.num_envs)
    env_terminals = np.zeros([len(rnn_states)], dtype=np.bool)
    vec_env.run_once = True

    # play the game...
    results = [(0, 0, 0) for _ in range(trials)]
    while not all(env_terminals):

        with torch.no_grad():
            model_output, new_rnn_states = algorithm.forward(env_obs, rnn_states)
            rnn_states[:] = new_rnn_states[:]

            log_policy = model_output["log_policy"].detach().cpu().numpy()
            actions = utils.sample_action_from_logp(log_policy)

        env_obs, env_rewards, env_terminals, env_infos = vec_env.step(actions)

        # look for finished games
        for i, env in enumerate(vec_env.games):
            if env.outcome != "":
                results[i] = env.team_scores

    # collate results
    red_score = np.mean([r for r, g, b in results])
    green_score = np.mean([g for r, g, b in results])
    blue_score = np.mean([b for r, g, b in results])

    return red_score, green_score, blue_score


def export_video(filename, algorithm: PMAlgorithm, scenario):
    """
    Exports a movie of model playing in given scenario
    """

    scale = 8

    # make a new environment so we don't mess the settings up on the one used for training.
    # it also makes sure that results from the video are not included in the log
    vec_env = make_env(scenario, parallel_envs=1, name="video")

    env_obs = vec_env.reset()
    env = vec_env.games[0]
    frame = env.render("rgb_array")

    obs_size = 39

    # work out our height
    height, width, channels = frame.shape
    if algorithm.enable_deception:
        height += len(env.players) * obs_size
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height), isColor=True)

    # this is required to make sure the last frame is visible
    vec_env.auto_reset = False

    rnn_state = algorithm.get_initial_state(len(env.players))

    # play the game...
    while env.outcome == "":

        with torch.no_grad():
            model_output, new_rnn_state = algorithm.forward(env_obs, rnn_state)

            rnn_state[:] = new_rnn_state[:]

            log_policy = model_output["log_policy"].detach().cpu().numpy()
            actions = utils.sample_action_from_logp(log_policy)

        env_obs, env_rewards, env_dones, env_infos = vec_env.step(actions)

        n_players = len(env.players)

        # format the role predictions
        # role predictions are in public_id order so that they change each round, we need to put them back
        # into index order.
        if config.enable_deception:

            # role prediction
            raw_predictions = model_output["role_prediction"].detach().cpu().numpy()
            role_predictions = np.zeros_like(raw_predictions)
            for i in range(n_players):
                for j in range(n_players):
                    role_predictions[i, j] = np.exp(raw_predictions[i, env.players[j].public_id])
        else:
            role_predictions = None

        # generate frames from global perspective
        frame = env.render("rgb_array", role_predictions=role_predictions)


        # overlay observation predictions
        if config.enable_deception:

            blank_frame = np.zeros([height, width, 3], dtype=np.uint8)
            blank_frame[:frame.shape[0], :frame.shape[1], :] = frame  # copy into potentially larger frame
            frame = blank_frame

            # observation frames are [n_players, n_players, h, w, c]
            obs_predictions = model_output["obs_prediction"].detach().cpu().numpy()
            obs_truth = env_obs.copy()

            # ground truth
            for i in range(n_players):
                dx = 0
                dy = height//2 + i * obs_size
                frame[dy:dy + obs_size, dx:dx + obs_size] = obs_truth[i, :, :, :3]

            for i in range(n_players):
                for j in range(n_players):
                    dx = j * obs_size + obs_size
                    dy = height//2 + i * obs_size
                    frame[dy:dy+obs_size, dx:dx+obs_size] = np.asarray(obs_predictions[i, j, :, :, :3]*255, dtype=np.uint8)

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
    scenario_descriptions = set(str(env.scenario) for env in vec_env.games)
    for description in scenario_descriptions:
        print(description)
    print()
    print("Config:")
    print(config)
    print()

    model = make_algo(vec_env)

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
        model.save(f"{config.log_folder}/model_{epoch:03}_M.pt")

        sub_epoch = 0

        step_counter = learn(model, step_counter, (epoch+1)*1e6, verbose=config.verbose == 1)
        print()

        # flush the log buffer and print scores
        rescue.flush_logs()
        print_scores(epoch=epoch)

    model.save(f"{config.log_folder}/model_final.p")
    if config.export_video:
        export_video(f"{config.log_folder}/ppo_run_{config.epochs:03}_M.mp4", model, config.train_scenarios[0])

    time_taken = time.time() - start_time
    print(f"Finished training after {time_taken/60/60:.1f}h.")

def make_algo(vec_env: MultiAgentVecEnv, model_name = None):

    algo_params = config.algo_params.copy()

    algo_params["model_name"] = model_name or config.model

    algorithm = PMAlgorithm(vec_env, device=config.device, amp=config.amp,
                            micro_batch_size=config.micro_batch_size, n_steps=config.n_steps,
                            enable_deception=config.enable_deception,
                            verbose=config.verbose >= 2, **algo_params)

    print(f" -model created using batch size of {algorithm.batch_size} and mini-batch size of"+
          f" {algorithm.mini_batch_size} with {algorithm.micro_batches} micro batch(es).")

    return algorithm

def run_benchmarks(train=True, model=True, env=True):

    def bench_scenario(scenario_name):
        """
        Evaluate how fast the scenarios ran
        :param scenario_name:
        :return:
        """
        vec_env = make_env(scenario_name, name="benchmark")
        _ = vec_env.reset()
        steps = 0
        start_time = time.time()
        while time.time() - start_time < 10:
            random_actions = np.random.randint(0, 10, size=[vec_env.num_envs])
            states, _, _, _ = vec_env.step(random_actions)
            steps += vec_env.num_envs
        torch.cuda.synchronize()
        time_taken = (time.time() - start_time)
        print(f" - scenario {scenario_name} runs at {steps / time_taken / 1000:.1f}k FPS.")

    def bench_training(scenario_name, model_name):
        """
        Evaluate how fast training runs
        :param scenario_name:
        :param model_name:
        :return:
        """
        vec_env = make_env(scenario_name, name="benchmark")
        algo = make_algo(vec_env, model_name)
        algo.learn(algo.batch_size) # just to warm it up
        start_time = time.time()
        algo.learn(2 * algo.batch_size)
        torch.cuda.synchronize()
        time_taken = (time.time() - start_time)
        print(f" -model {model_name} trains at {utils.Color.WARNING}{2 * algo.batch_size / time_taken / 1000:.1f}k"+
              f"{utils.Color.ENDC} FPS.")

    def bench_model(model_name):
        """
        Evaluate the inference time of the model (without training)
        :param model_name:
        :return:
        """
        vec_env = make_env("red2")
        agent = make_algo(vec_env, model_name)
        obs = np.asarray(vec_env.reset())
        steps = 0
        start_time = time.time()
        while time.time() - start_time < 10:

            with torch.no_grad():
                model_output, _ = agent.forward(obs, agent.agent_rnn_state)
                log_policy = model_output["log_policy"].detach().cpu().numpy()
                actions = utils.sample_action_from_logp(log_policy)
            steps += vec_env.num_envs
        torch.cuda.synchronize()
        time_taken = (time.time() - start_time)
        print(f" - model {model_name} runs at {steps / time_taken / 1000:.1f}k FPS.")

    if train:
        print("Benchmarking training...")
        for model_name in ["default", "fast", "global"]:
            bench_training("red2", model_name)

    if env:
        print("Benchmarking environments...")
        for scenario_name in ["full", "medium", "red2"]:
            bench_scenario(scenario_name)

    if model:
        print("Benchmarking models (inference)...")
        for model_name in ["default", "fast"]:
            bench_model(model_name)

def print_scores(epoch=None):
    """ Print current scores, also makes a plot"""
    try:
        log_file = os.path.join(config.log_folder, "env_0.csv")
        results = load_results(log_file)
        scores = tuple(round(get_score(results, team), 1) for team in ["red", "green", "blue"])
        print(f" -training scores: {scores}")

        teams = set()

        for scenario in config.train_scenarios + config.eval_scenarios:
            for id, team in enumerate(["red", "green", "blue"]):
                if scenario.team_counts[id] > 0:
                    teams.add(team)

        export_graph(log_file, epoch=epoch, png_base_name="train", teams=teams)
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
    model = PMAlgorithm.load(filename, env)
    return model

def learn(agent: MarlAlgorithm, step_counter, max_steps, verbose=True):

    sub_epoch = 0

    while step_counter < max_steps:

        global CURRENT_EPOCH
        CURRENT_EPOCH = step_counter / 1e6

        # learn will round down to nearest batch. Our batches are often around 49k, so running
        # .learn(100000) will only actually generate 88k steps. To keep all the numbers correct I therefore
        # run each batch individually. For large batch sizes this should be fine.
        learn_steps = agent.batch_size
        start_epoch_time = time.time()
        agent.learn(learn_steps, reset_num_timesteps=step_counter == 0)
        step_counter += learn_steps
        epoch_time = time.time() - start_epoch_time

        fps = learn_steps / epoch_time

        # ignore first sub-epoch as FPS will be lower than normal
        if verbose:
            if sub_epoch == 1:
                print(f" -FPS: {utils.Color.OKGREEN}{fps:.0f}{utils.Color.ENDC} .", end='', flush=True)
            elif sub_epoch > 1:
                print(".", end='', flush=True)

        # this is needed to stop a memory leak
        gc.collect()

        sub_epoch += 1

    return step_counter

def run_test(scenario_name, n_steps=int(2e6)):

    destination_folder = os.path.join(config.log_folder, scenario_name)
    os.makedirs(destination_folder, exist_ok=True)
    log_file = os.path.join(destination_folder, "env_0.csv")

    # our MARL environments are handled like vectorized environments
    make_env = lambda: RescueTheGeneralEnv(scenario_name=scenario_name, name="test", log_file=log_file)
    vec_env = MultiAgentVecEnv([make_env for _ in range(config.parallel_envs)])

    model = make_algo(vec_env)

    learn(model, 0, n_steps, verbose=config.verbose == 1)
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

def regression_test():

    print("Performing regression test, this could take some time.")

    start_time = time.time()

    # copy in files
    # todo, this should be a function...
    from shutil import copyfile
    for filename in ["train.py", "rescue.py"]:
        copyfile(filename, f"{config.log_folder}/{filename}")
    with open(f"{config.log_folder}/config.txt", "w") as f:
        f.write(str(config))

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

def quick_test():

    print("Performing regression test, this could take some time.")

    start_time = time.time()

    # copy in files
    # todo, this should be a function...
    from shutil import copyfile
    for filename in ["train.py", "rescue.py"]:
        copyfile(filename, f"{config.log_folder}/{filename}")
    with open(f"{config.log_folder}/config.txt", "w") as f:
        f.write(str(config))

    for scenario_name, team, required_score in [
        ("red2", "red", 7.5)
    ]:

        results = run_test(scenario_name)
        score = get_score(results, team)
        score_alt = get_score_alt(results, team)

        result = "FAIL" if score < required_score else "PASS"
        print(f"  [{result:}] {scenario_name:<20} ({score:.1f}, {score_alt:.1f})")

    time_taken = time.time() - start_time
    print(f"Finished tests in {time_taken/60:.1f}m.")


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def profile():

    print("Profiling model")

    vec_env = make_env("red2", name="profile")
    algo = make_algo(vec_env)
    algo.learn(algo.batch_size)  # just to warm it up

    # run profiler
    with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
        algo.learn(algo.batch_size)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # get a trace
    with profiler.profile() as prof:
        with profiler.record_function("train_step"):
            algo.learn(algo.batch_size)
    prof.export_chrome_trace("trace.json")

    print("done.")



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="[benchmark|train|test|evaluate]")
    parser.add_argument('--run', type=str, help="run folder", default="test")
    parser.add_argument('--device', type=str, help="[CPU|AUTO|CUDA|CUDA:n]", default="auto")

    parser.add_argument('--train_scenarios', type=str, default="full",
        help="Scenario settings for training. Can be a single scenario name e.g. 'red2' or for a mixed training setting "
             +" use [['<scenario>', '<red_strat>', '<green_strat>', '<blue_strat>'], ...]")
    parser.add_argument('--eval_scenarios', type=str,
        help="Scenario settings used to evaluate (defaults to same as train_scenario)", default=None)

    parser.add_argument('--epochs', type=int, help="number of epochs to train for (each 1M agent steps)", default=500)
    parser.add_argument('--script_blue_team', type=str, default=None)
    parser.add_argument('--export_video', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--algo_params', type=str, default="{}")
    parser.add_argument('--verbose', type=int, default=1, help="Level of logging output, 0=off, 1=normal, 2=full.")

    parser.add_argument('--parallel_envs', type=int, default=128,
                        help="The number of times to duplicate the environments. Note: when using mixed learning each"+
                             "environment will be duplicated this number of times.")

    parser.add_argument('--n_steps', type=int, default=128)

    parser.add_argument('--vary_team_player_counts', type=str2bool, nargs='?', const=True,  default=False, help="Use a random number of players turning training.")

    parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False,
                        help="Enable Automatic Mixed Precision")

    parser.add_argument('--enable_deception', type=str2bool, nargs='?', const=True, default=False,
                        help="Enables the deception module during training")

    parser.add_argument('--model', type=str, help="model to use [default|fast]", default="default")
    parser.add_argument('--data_parallel', type=str2bool, nargs='?', const=True, default=False,
                        help="Enable data parallelism, can speed things up on multi-gpu machines.")
    parser.add_argument('--micro_batch_size', type=str, default="auto",
                        help="Number of samples per micro-batch, reduce if GPU ram is exceeded.")



    args = parser.parse_args()
    config.setup(args)

    print()
    print(f"Starting {config.log_folder} on device {config.device}")

    os.makedirs(config.log_folder, exist_ok=True)

    if args.mode == "benchmark":
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
    elif args.mode == "profile":
        profile()
    elif args.mode == "quick_test":
        quick_test()
    else:
        raise Exception(f"Invalid mode {args.mode}")


if __name__ == "__main__":

    CURRENT_EPOCH = 0
    RescueTheGeneralEnv.get_current_epoch = get_current_epoch
    config = Config()
    main()