"""
Simple hyper-parameter search
"""

import multiprocessing
import os
import random

def run_experiment(job):

    print("Starting ", multiprocessing.current_process()._identity[0])

    id, params = job
    device = multiprocessing.current_process()._identity[0] % 4

    run_name = f"search/{id}"

    if os.path.exists(run_name):
        return

    # stub, should be train, not test with --train_scenarios='red2'

    script = \
        f"python train.py test --run=\"{run_name}\" --device=cuda:{device} --epochs=2 " + params

    print()
    print(script)
    print()

    os.system(script)


if __name__ == "__main__":

    pool = multiprocessing.Pool(processes=4)

    jobs = []
    id = 0

    for i in range(100):

        n_steps = random.choice([32, 64, 128])
        learning_rate = random.choice([1e-4, 2.5e-4, 1e-3])
        model = random.choice(["default", "fast"])
        parallel_envs = random.choice([32, 64, 128, 256])
        entropy_bonus = random.choice([0.003, 0.01, 0.03])
        mini_batches = random.choice([4, 8, 16])
        adam_epsilon = random.choice([1e-5, 1e-8])
        memory_units = random.choice([32, 64, 128, 256, 512])
        out_features = random.choice([32, 64, 128, 256, 512])
        max_grad_norm = random.choice([None, 0.5, 5.0])
        gamma = random.choice([0.9, 0.95, 0.99, 0.995])

        algo_params = "{"+\
                      f"'n_steps':{n_steps}, "+ \
                      f"'learning_rate':{learning_rate}, " + \
                      f"'adam_epsilon':{adam_epsilon}, " + \
                      f"'gamma':{gamma}, " + \
                      f"'entropy_bonus':{entropy_bonus}, " + \
                      f"'mini_batches':{mini_batches}, " + \
                      f"'memory_units':{memory_units}, " + \
                      f"'out_features':{out_features}, " + \
                      f"'max_grad_norm':{max_grad_norm}" + \
                "}"

        params = f"--model={model} --parallel_envs={parallel_envs} --algo_params=\"{algo_params}\""

        jobs.append([id, params])
        id += 1

    random.shuffle(jobs)

    pool.map(run_experiment, jobs)