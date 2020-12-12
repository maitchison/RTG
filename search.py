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

    script = \
        f"python ./run/{RUN_NAME}/train.py train --run=\"{run_name}\" --device={device} --epochs=20 " + params

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

    os.system(f'mkdir ./run/{args.run}')
    os.system(f'cp *.py ./run/{args.run}')

    for i in range(256):

        # search 3
        # n_steps = random.choice([8, 16, 32, 64, 128])
        # learning_rate = random.choice([1e-4, 2.5e-4, 1e-3])
        # model = random.choice(["default", "large"])
        # parallel_envs = random.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
        # entropy_bonus = random.choice([0.003, 0.01, 0.03])
        # mini_batches = random.choice([4, 8, 16])
        # adam_epsilon = random.choice([1e-5, 1e-8])
        # memory_units = random.choice([32, 64, 128, 256, 512])
        # out_features = random.choice([32, 64, 128, 256, 512])
        # max_grad_norm = random.choice([None, 0.5, 5.0])
        # gamma = random.choice([0.95, 0.99, 0.995])
        # amp = random.choice([False])

        # search 4
        # the idea here is to see if we can do very large parallel envs, and at the same time
        # increase the test length to 4 epochs (as I suspect they will train slower but better)
        # I'm also interested in finding out how mini_batches works now that I removed loss
        # attenuation during microbatches
        # n_steps = random.choice([32]) # shortest one that works
        # learning_rate = random.choice([2.5e-4])
        # model = random.choice(["default"])
        # parallel_envs = random.choice([2**x for x in [7, 8, 9, 10, 11, 12, 13, 14]])
        # entropy_bonus = random.choice([0.01])
        # mini_batches = random.choice([4, 8, 16])
        # adam_epsilon = random.choice([1e-8])
        # memory_units = random.choice([256])
        # out_features = random.choice([256])
        # max_grad_norm = random.choice([5.0])
        # gamma = random.choice([0.99])
        # amp = random.choice([False])
        #
        # search 5
        # / n_minibatches is back
        # trying really hard to get parallel envs higher
        # n_steps = random.choice([32])  # shortest one that works
        # learning_rate = random.choice([2.5e-4])
        # model = random.choice(["default"])
        # parallel_envs = random.choice([2 ** x for x in [13, 14, 15, 16]])
        # entropy_bonus = random.choice([0.01])
        # mini_batches = random.choice([8])
        # adam_epsilon = random.choice([1e-8])
        # memory_units = random.choice([256])
        # out_features = random.choice([256])
        # max_grad_norm = random.choice([5.0])
        # gamma = random.choice([0.99])
        # amp = random.choice([False])

        # main_params = f"--amp={amp} --model={model} --n_steps={n_steps} --parallel_envs={parallel_envs} --test_epochs=4"
        #
        # algo_params = "{"+\
        #               f"'learning_rate':{learning_rate}, " + \
        #               f"'adam_epsilon':{adam_epsilon}, " + \
        #               f"'gamma':{gamma}, " + \
        #               f"'entropy_bonus':{entropy_bonus}, " + \
        #               f"'mini_batches':{mini_batches}, " + \
        #               f"'memory_units':{memory_units}, " + \
        #               f"'out_features':{out_features}, " + \
        #               f"'max_grad_norm':{max_grad_norm}" + \
        #         "}"

        # prediction_search_1

        dm_max_window_size = random.choice([8, 16, 32])
        dm_replay_buffer_multiplier = random.choice([1, 4, 8])
        dm_mini_batches = random.choice([4, 8, 16, 32, 64])
        dm_out_features = random.choice([512, 1024, 2048])
        dm_memory_units = random.choice([512, 1024, 2048])
        dm_lstm_mode = random.choice(['off', 'on', 'residual', 'cat'])
        dm_xy_factor = random.choice([1, 0.1, 0.01, 0])
        dm_learning_rate = random.choice([3e-3, 1e-3, 2.5e-4])
        dm_loss_fn = random.choice(["mse", "l1", "l2"])
        dm_loss_scale = random.choice([1, 10, 100])

        main_params = "--train_scenario='r2g2_hrp' --enable_deception=True "

        algo_params = "{" +\
                      f"'dm_max_window_size':{dm_max_window_size}," + \
                      f"'dm_replay_buffer_multiplier':{dm_replay_buffer_multiplier}," + \
                      f"'dm_mini_batches':{dm_mini_batches}," + \
                      f"'dm_memory_units':{dm_memory_units}," + \
                      f"'dm_out_features':{dm_out_features}," + \
                      f"'dm_lstm_mode':'{dm_lstm_mode}'," + \
                      f"'dm_xy_factor':{dm_xy_factor}," + \
                      f"'dm_loss_fn':'{dm_loss_fn}'," + \
                      f"'dm_loss_scale':{dm_loss_scale}," + \
                      f"'dm_learning_rate':{dm_learning_rate}," + \
                      "}"

        params = f"{main_params} --algo_params=\"{algo_params}\""

        jobs.append([id, params])
        id += 1

    pool.map(run_experiment, jobs)