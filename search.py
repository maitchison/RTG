"""
Simple hyper-parameter search
"""

import multiprocessing
import os

def run_experiment(job):

    id, n_step, learning_rate, model, n_env = job
    device = id % 4

    run_name = f"search/{id}"

    params = {
        'learning_rate': learning_rate,
        'n_steps': n_step,
    }

    if os.path.exists(run_name):
        return

    script = f"""python train.py train --model={model} --parallel_envs={n_env} --algo_params="{params}" --scenario='red2' --run={run_name} --device={device} --epochs=2"""

    print()
    print(script)
    print()

    os.system(script)


if __name__ == "__main__":

    pool = multiprocessing.Pool(processes=8)

    # 3*3*2*4 = 72, can do 16 an hour so 4.5 hours if no crash

    n_steps_values = [40, 64, 128]
    learning_rate_values = [1e-4, 2.5e-4, 1e-3]
    model_values = ["cnn_lstm_default", "cnn_lstm_fast"]
    n_envs_values = [8, 16, 32, 64]

    jobs = []
    id = 0

    for n_step in n_steps_values:
        for learning_rate in learning_rate_values:
            for model in model_values:
                for n_env in n_envs_values:
                    jobs.append([id, n_step, learning_rate, model, n_env])
                    id += 1

    pool.map(run_experiment, jobs)
