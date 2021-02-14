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

    id, params, name, devices, run = job

    device = devices[multiprocessing.current_process()._identity[0]-1]

    path = f"{run}/{name}"

    if has_been_started(os.path.join('run',run), name):
        print(f"skipping {path} as it exists.")
        return

    script = \
        f"python ./run/{run}/train.py train --run=\"{path}\" --device={device} " + params

    print()
    print(script)
    print()

    os.system(script)

def quote_if_str(x):
    return f"'{x}'" if type(x) is str else x

def algo_dict_to_str(params):
    return "{" + (", ".join([f"'{k}':{quote_if_str(v)}" for k, v in params.items()])) + "}"

def random_search(main_params, search_params, count=128):

    jobs = []
    id = 0
    for i in range(count):
        params = {}
        for k, v in search_params.items():
            params[k] = random.choice(v)

        # build the string
        algo_params = {}
        normal_params = {}
        for k, v in params.items():
            if k.startswith("a:"):
                algo_params[k[2:]] = v
            else:
                normal_params[k] = v

        for model_code in ['dm', 'pm']:
            if algo_params.get(f'{model_code}_lstm_mode', 'residual') == 'residual' and f"{model_code}_memory_units" in algo_params:
                # in residual mode out features must match memory units
                algo_params[f'{model_code}_out_features'] = algo_params[f"{model_code}_memory_units"]

        normal_params = " ".join(f"--{k}={quote_if_str(v)}" for k, v in normal_params.items())
        algo_params = algo_dict_to_str(algo_params)

        params = f"{main_params} {normal_params} --algo_params=\"{algo_params}\""

        jobs.append([id, params, f"{args.run}-{str(id)}", DEVICES, args.run])
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

            if params.get('dm_lstm_mode', 'residual') == 'residual' and k == 'dm_memory_units':
                # in residual mode out features must match memory units
                params['dm_out_features'] = v

            if params.get('lstm_mode', 'residual') == 'residual' and k == 'memory_units':
                # in residual mode out features must match memory units
                params['out_features'] = v

            key_name = k[2:] if k.startswith("a:") else k
            name = f"{key_name}-{v}-1"

            # build the string
            algo_params = {}
            normal_params = {}
            for k, v in params.items():
                if k.startswith("a:"):
                    algo_params[k[2:]] = v
                else:
                    normal_params[k] = v

            normal_params = " ".join(f"--{k}={quote_if_str(v)}" for k, v in normal_params.items())
            algo_params = algo_dict_to_str(algo_params)

            params = f"{main_params} {normal_params} --algo_params=\"{algo_params}\""
            jobs.append([id, params, name, DEVICES, args.run])
            id += 1

    return jobs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default="['cuda']")
    parser.add_argument('--run', type=str, default="search_dm")
    args = parser.parse_args()

    DEVICES = ast.literal_eval(args.devices)

    print("Training on devices", DEVICES)

    pool = multiprocessing.Pool(processes=len(DEVICES))

    destination_path = os.path.join("run", args.run)
    os.system(f'mkdir {destination_path}')


    # copy source files so that modifications will not effect search
    if not os.path.exists(os.path.join(destination_path, "train.py")):
        import platform
        print("Copying train.py")
        if platform.system() == "Windows":
            os.system(f'copy *.py {destination_path}')
        else:
            os.system(f'cp *.py {destination_path}')

    else:
        print("Using previous train.py")

    main_params = " ".join([
        "--train_scenarios=r2g2_hr",
        "--eval_scenarios=[]",
        "--micro_batch_size=512",  # 1024 is slower, but can run two searches in parallel
        "--prediction_mode=action",
        "--verbose=2",
        "--deception_bonus=[0,0,0]",
        "--save_model=none",
        "--epochs=25"
    ])

    # search_params = {
    #     'n_steps': [16, 32, 64, 128],
    #     'parallel_envs': [128, 256, 512, 1024],
    #     'a:learning_rate': [1e-4, 2.5e-4, 1e-3],
    #     'a:pm_lstm_mode': ['off', 'on', 'cat', 'residual'],
    #     'a:pm_memory_units': [64, 128, 256, 512, 1024],
    #     'a:pm_out_features': [64, 128, 256, 512, 1024],
    #     'a:gamma': [0.95, 0.99, 0.995],
    #     'a:entropy_bonus': [0.003, 0.01, 0.03],
    #     'a:mini_batch_size': [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    #     'a:adam_epsilon': [1e-5, 1e-8],
    #     'a:max_grad_norm': [None, 0.5, 5.0]
    # }


    search_params = {
        'a:dm_mini_batch_size': [64, 128, 256, 512, 1024],
        'a:dm_memory_units': [256, 512, 1024, 2048, 4096],
        'a:dm_lstm_mode': ['residual'],
        'a:dm_loss_scale': [0.1, 1, 10],
        'a:dm_max_window_size': [4, 8, None],
        'n_steps': [16],
        'parallel_envs': [512],
        'a:learning_rate': [2.5e-4],
        'a:pm_lstm_mode': ['residual'],
        'a:pm_memory_units': [512],
        'a:gamma': [0.99],
        'a:entropy_bonus': [0.003, 0.01, 0.03],
        'a:mini_batch_size': [2048],
        'a:adam_epsilon': [1e-8],
        'a:max_grad_norm': [0.5]
    }


    jobs = random_search(main_params, search_params)
    print()
    print("-"*60)
    print()
    for job in jobs:
        print(job)
    print()
    print("-" * 60)
    print()

    pool.map(run_experiment, jobs)

    # python search.py --device="['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']"
    # python search.py - -device = "['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']"
    # python search.py --device="['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1']"