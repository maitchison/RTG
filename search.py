"""
Simple hyper-parameter search
"""

import multiprocessing
import os
import random
import argparse
import ast

def has_been_started(base_path, run):
    folders = [x[0] for x in os.walk(base_path)]
    return any(run in folder for folder in folders)

def run_experiment(job):

    id, params, name = job

    device = DEVICES[multiprocessing.current_process()._identity[0]-1]

    path = f"{args.run}/{name}"

    if has_been_started(os.path.join('run',args.run), name):
        print(f"skipping {path} as it exists.")
        return

    # todo: look for and ignore folders that are already done

    script = \
        f"python ./run/{args.run}/train.py train --run=\"{path}\" --device={device} " + params

    print()
    print(script)
    print()

    os.system(script)

def quote_if_str(x):
    return f"'{x}'" if type(x) is str else x

def algo_dict_to_str(params):
    return "{" + (" ".join([f"'{k}':{quote_if_str(v)}" for k, v in params.items()])) + "}"

def random_search(main_params, search_params, count=256):

    jobs = []
    id = 0
    for i in range(count):
        params = {}
        for k, v in search_params.items():
            params[k] = random.choice(v)

        algo_params = algo_dict_to_str(params)

        params = f"{main_params} --algo_params=\"{algo_params}\""

        jobs.append([id, params, str(id)])
        id += 1

    return jobs

def axis_search(main_params, search_params, default_params=None):

    jobs = []
    id = 0

    for k, values in search_params.items():
        for v in values:
            if default_params is not None:
                params = default_params.copy()
            else:
                params = {}
            params[k] = v

            name = f"{k}-{v}-1"

            algo_params = algo_dict_to_str(params)
            params = f"{main_params} --algo_params=\"{algo_params}\""
            jobs.append([id, params, name])
            id += 1

    return jobs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default="['cuda']")
    parser.add_argument('--run', type=str, default="search_x")
    args = parser.parse_args()

    DEVICES = ast.literal_eval(args.devices)

    print("Training on devices", DEVICES)

    pool = multiprocessing.Pool(processes=len(DEVICES))

    os.system(f'mkdir ./run/{args.run}')

    # copy source files so that modifications will not effect search
    if not os.path.exists(f"./run/{args.run}/train.py"):
        print("Copying train.py")
        os.system(f'cp *.py ./run/{args.run}')
    else:
        print("Using previous train.py")

    main_params = " ".join([
        "--train_scenarios=r2g2_hrp",
        "--eval_scenarios=[]",
        "--micro_batch_size=1024",  # 1024 is slower, but can run two searches in parallel
        "--prediction_mode=others",
        "--save_model=none",
        "--epochs=25"
    ])

    search_params = {
                  'dm_max_window_size': [1, 2, 4, 8, 16, 32],
                  'dm_replay_buffer_multiplier': [1, 2, 4, 8, 16],
                  'dm_mini_batch_size': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                  'dm_memory_units': [256, 512, 1024, 2048],
                  'dm_lstm_mode': ['off', 'on', 'residual', 'cat'],
                  'dm_kl_factor': [0, 0.5],
                  'dm_vision_filter': [0, 0.5, 1],
                  'dm_loss_scale': [0.001, 0.1, 1, 10, 100, 1000],
                  'dm_learning_rate': [1e-3, 2.5e-4, 1e-4]
    }

    jobs = axis_search(main_params, search_params)
    print()
    print("-"*60)
    print()
    for job in jobs:
        print(job)
    print()
    print("-" * 60)
    print()
    pool.map(run_experiment, jobs)
    # stub: new version
