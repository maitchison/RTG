"""
Train agents in the Rescue the General Game

Author: Matthew Aitchison

"""

import torch

# this dramatically reduces the CPU resourced used and makes no appreciable
# difference to performance
torch.set_num_threads(2)

import torch.cuda

import numpy as np
import argparse
import time
import gc
import os

import rescue
import utils
import video

from utils import Color as C
from support import make_env, evaluate_model, Config, make_algo

from rescue import RescueTheGeneralEnv
from marl_env import MultiAgentVecEnv
from tools import load_results, get_score, get_score_alt, export_graph
from algorithms import PMAlgorithm, MarlAlgorithm
from typing import Union
import torch.autograd.profiler as profiler

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

    vec_env = make_env(config.train_scenarios, config.parallel_envs, name="train", log_path=config.log_folder)

    print("Scenario parameters:")
    scenario_descriptions = set(str(env.scenario) for env in vec_env.games)
    for description in scenario_descriptions:
        print(description)
    print()
    print("Config:")
    print(config)
    print()

    algorithm = make_algo(vec_env, config)

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

            scores = evaluate_model(algorithm, eval_scenario, sub_folder, trials=100)
            rounded_scores = tuple(round(float(score), 1) for score in scores)

            print(f" -evaluation against {str(eval_scenario):<40} {rounded_scores}")

            # generate a video
            if config.export_video:
                video.export_video(f"{sub_folder}/evaluation_{epoch:03}_M.mp4", algorithm, eval_scenario)

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
            video.export_video(f"{config.log_folder}/training_{epoch:03}_M.mp4", algorithm, config.train_scenarios[0])

        # save model
        if config.save_model == "all":
            algorithm.save(f"{config.log_folder}/model_{epoch:03}_M.pt")
        elif config.save_model == "none":
            pass
        elif config.save_model == "recent":
            algorithm.save(f"{config.log_folder}/model.pt")
        else:
            try:
                save_every = int(config.save_model)
                if epoch % save_every == 0:
                    algorithm.save(f"{config.log_folder}/model_{epoch:03}_M.pt")
            except:
                raise ValueError("Invalid save model parameter, use [none|recent|all|0..n].")
            algorithm.save(f"{config.log_folder}/model.pt")

        step_counter = learn(algorithm, step_counter, (epoch+1)*1e6, verbose=config.verbose == 1)
        print()

        # save logs
        algorithm.save_logs(config.log_folder)

        # flush the log buffer and print scores
        rescue.flush_logs()
        print_scores(epoch=epoch)

    algorithm.save(f"{config.log_folder}/model_final.p")
    if config.export_video:
        video.export_video(f"{config.log_folder}/ppo_run_{config.epochs:03}_M.mp4", algorithm, config.train_scenarios[0])

    time_taken = time.time() - start_time
    print(f"Finished training after {time_taken/60/60:.1f}h.")

def run_benchmarks(train=True, model=True, env=True):

    def bench_scenario(scenario_name):
        """
        Evaluate how fast the scenarios ran
        :param scenario_name:
        :return:
        """
        vec_env = make_env(scenario_name, config.parallel_envs, name="benchmark")
        _ = vec_env.reset()
        steps = 0
        start_time = time.time()
        while time.time() - start_time < 10:
            random_actions = np.random.randint(0, vec_env.action_space.n-1, size=[vec_env.num_envs])
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
        vec_env = make_env(scenario_name, config.parallel_envs, name="benchmark")
        algo = make_algo(vec_env, config, model_name)
        algo.learn(algo.batch_size) # just to warm it up
        start_time = time.time()
        algo.learn(2 * algo.batch_size)
        torch.cuda.synchronize()
        time_taken = (time.time() - start_time)
        print(f" -model {model_name} trains at {C.WARNING}{2 * algo.batch_size / time_taken / 1000:.1f}k"+
              f"{C.ENDC} FPS.")

    def bench_model(model_name):
        """
        Evaluate the inference time of the model (without training)
        :param model_name:
        :return:
        """
        vec_env = make_env("red2", config.parallel_envs)
        agent = make_algo(vec_env, config, model_name)
        obs = np.asarray(vec_env.reset())
        steps = 0
        start_time = time.time()

        while time.time() - start_time < 10:

            with torch.no_grad():
                roles = vec_env.get_roles()
                model_output, _ = agent.forward(
                    torch.from_numpy(obs),
                    agent.agent_rnn_state,
                    torch.from_numpy(roles)
                )
                log_policy = model_output["log_policy"].detach().cpu().numpy()
                actions = utils.sample_action_from_logp(log_policy)
            steps += vec_env.num_envs
        torch.cuda.synchronize()
        time_taken = (time.time() - start_time)
        print(f" - model {model_name} runs at {steps / time_taken / 1000:.1f}k FPS.")

    if train:
        print("Benchmarking training...")
        for model_name in ["default", "fast"]:
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

    last_log_save = 0

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
                print(f" -FPS: {C.OKGREEN}{fps:.0f}{C.ENDC} .", end='', flush=True)
            elif sub_epoch > 1:
                print(".", end='', flush=True)

        if time.time() - last_log_save > 5*60:
            # save logs every 5 minutes
            agent.save_logs(config.log_folder)

        # this is needed to stop a memory leak
        gc.collect()

        sub_epoch += 1

    return step_counter

def run_test(scenario_name, team, epochs=2):

    destination_folder = os.path.join(config.log_folder, scenario_name)
    os.makedirs(destination_folder, exist_ok=True)
    log_file = os.path.join(destination_folder, "env_0.csv")
    eval_log_file = os.path.join(destination_folder+"/eval", "env_0.csv")

    # our MARL environments are handled like vectorized environments
    make_env = lambda: RescueTheGeneralEnv(scenario_name, config.parallel_envs, name="test", log_file=log_file)
    vec_env = MultiAgentVecEnv([make_env for _ in range(config.parallel_envs)])

    algorithm = make_algo(vec_env, config)

    step_counter = 0
    for epoch in range(epochs):
        scores = evaluate_model(algorithm, scenario_name, f"{destination_folder}/eval", trials=100)
        if epoch != 0:
            print()

        # flush the log buffer
        rescue.flush_logs()
        results = load_results(eval_log_file)
        epoch_score = get_score(results, team)

        print(f" -eval_{epoch}: [{epoch_score:.1f}]", end='')
        step_counter = learn(algorithm, step_counter, (epoch + 1) * 1e6, verbose=config.verbose == 1)

    print()
    scores = evaluate_model(algorithm, scenario_name, f"{destination_folder}/eval", trials=100)
    rescue.flush_logs()
    results = load_results(eval_log_file)
    final_score = get_score(results, team)
    print(f" -final_eval: {final_score}")

    video.export_video(f"{destination_folder}/{scenario_name}.mp4", algorithm, scenario_name)

    try:
        export_graph(eval_log_file, epoch=epochs, png_base_name="results")
    except Exception as e:
        # not worried about this not working...
        print(e)

    # return scores
    return results

def regression_test(tests: Union[str, tuple, list] = ("mem2b", "red2", "green2", "blue2")):

    print(f"Performing regression tests on {config.test_epochs} epochs, this could take some time.")

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
        if scenario_name not in tests:
            continue

        results = run_test(scenario_name, team, config.test_epochs)
        score = get_score(results, team)
        score_alt = get_score_alt(results, team)

        if score < required_score:
            result = f"{C.FAIL}FAIL{C.ENDC}"
        else:
            result = f"{C.OKGREEN}PASS{C.ENDC}"
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

    vec_env = make_env(config.eval_scenarios, config.parallel_envs, name="profile")
    algo = make_algo(vec_env, config)
    # get a trace
    with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
        with profiler.record_function("train_step"):
            algo.learn(algo.batch_size)
    prof.export_chrome_trace("trace.json")

    print("done.")



def main():

    print("CUDA:", torch.version.cuda)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="[benchmark|train|test|evaluate]")
    parser.add_argument('--run', type=str, help="run folder", default="test")
    parser.add_argument('--device', type=str, help="[CPU|AUTO|CUDA|CUDA:n]", default="auto")
    parser.add_argument('--deception_bonus', type=str, help="Bonus to give agents for applying deception (per team)", default=0)

    parser.add_argument('--train_scenarios', type=str, default="rescue",
        help="Scenario settings for training. Can be a single scenario name e.g. 'red2' or for a mixed training setting "
             +" use [['<scenario>', '<red_strat>', '<green_strat>', '<blue_strat>'], ...]")
    parser.add_argument('--eval_scenarios', type=str,
        help="Scenario settings used to evaluate (defaults to same as train_scenario)", default=None)

    parser.add_argument('--epochs', type=int, help="number of epochs to train for (each 1M agent steps)", default=500)
    parser.add_argument('--test_epochs', type=int, help="number of epochs to test for during test mode (each 1M agent steps)", default=2)

    parser.add_argument('--script_blue_team', type=str, default=None)
    parser.add_argument('--export_video', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--algo_params', type=str, default="{}")
    parser.add_argument('--verbose', type=int, default=1, help="Level of logging output, 0=off, 1=normal, 2=full.")
    parser.add_argument('--save_model', type=str, default="10", help="Enables model saving, [all|0..n|recent|none].")

    parser.add_argument('--n_steps', type=int, default=16)
    parser.add_argument('--max_window_size', type=int, default=None)

    parser.add_argument('--use_global_value', type=str2bool, nargs='?', const=True, default=False,
                        help="Enable Global value function")

    parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False,
                        help="Enable Automatic Mixed Precision")
    parser.add_argument('--parallel_envs', type=int, default=512,
                        help="The number of times to duplicate the environments. Note: when using mixed learning each"+
                             "environment will be duplicated this number of times.")
    parser.add_argument('--model', type=str, help="model to use [default|fast]", default="default")

    parser.add_argument('--export_rollout', type=str2bool, nargs='?', const=True, default=False,
                        help="Exports rollout to disk, very large")

    parser.add_argument('--prediction_mode', type=str, default="off", help="off|action|observation|both")

    parser.add_argument('--micro_batch_size', type=str, default="auto",
                        help="Number of samples per micro-batch, reduce if GPU ram is exceeded.")

    parser.add_argument('--split_policy', type=str2bool, nargs='?', const=True, default=False,
                        help="Uses separate models for each role (slower).")

    parser.add_argument('--nan_check', type=str2bool, nargs='?', const=True, default=False,
                        help="Check for nans / extreme values in output (slower).")

    args = parser.parse_args()
    config.setup(vars(args))

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
    elif args.mode == "memory_test":
        regression_test("mem2b")
    elif args.mode == "red_test":
        regression_test("red2")
    elif args.mode == "green_test":
        regression_test("green2")
    elif args.mode == "blue_test":
        regression_test("blue2")
    else:
        raise Exception(f"Invalid mode {args.mode}")


if __name__ == "__main__":
    CURRENT_EPOCH = 0
    RescueTheGeneralEnv.get_current_epoch = get_current_epoch
    config = Config()
    main()