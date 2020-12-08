"""
Simple hyper-parameter search
"""

import multiprocessing
import os
import random
import argparse
import ast

def run_experiment(job):

    id, params = job

    device = DEVICES[multiprocessing.current_process()._identity[0]-1]

    run_name = f"{RUN_NAME}/{id}"

    if os.path.exists(run_name):
        print(f"skipping {run_name} as it exists.")
        return

    # stub, should be train, not test with --train_scenarios='red2'

    script = \
        f"python train.py test --run=\"{run_name}\" --device={device} " + params

    print()
    print(script)
    print()

    os.system(script)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default="['cuda']")
    parser.add_argument('--run', type=str, default="search_x")
    args = parser.parse_args()

    DEVICES = ast.literal_eval(args.devices)
    RUN_NAME = args.run

    print("Training on devices", DEVICES)

    pool = multiprocessing.Pool(processes=len(DEVICES))

    jobs = []
    id = 0

    for i in range(256):

        n_steps = random.choice([8, 16, 32, 64, 128])
        learning_rate = random.choice([1e-4, 2.5e-4, 1e-3])
        model = random.choice(["default"])
        parallel_envs = random.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
        entropy_bonus = random.choice([0.003, 0.01, 0.03])
        mini_batches = random.choice([4, 8, 16])
        adam_epsilon = random.choice([1e-5, 1e-8])
        memory_units = random.choice([32, 64, 128, 256, 512])
        out_features = random.choice([32, 64, 128, 256, 512])
        max_grad_norm = random.choice([None, 0.5, 5.0])
        gamma = random.choice([0.95, 0.99, 0.995])
        amp = random.choice([False])

        main_params = f"--amp={amp} --model={model} --n_steps={n_steps} --parallel_envs={parallel_envs}"

        algo_params = "{"+\
                      f"'learning_rate':{learning_rate}, " + \
                      f"'adam_epsilon':{adam_epsilon}, " + \
                      f"'gamma':{gamma}, " + \
                      f"'entropy_bonus':{entropy_bonus}, " + \
                      f"'mini_batches':{mini_batches}, " + \
                      f"'memory_units':{memory_units}, " + \
                      f"'out_features':{out_features}, " + \
                      f"'max_grad_norm':{max_grad_norm}" + \
                "}"

        params = f"{main_params} --algo_params=\"{algo_params}\""

        jobs.append([id, params])
        id += 1

    pool.map(run_experiment, jobs)