import time
import numpy as np
import gzip
import pickle
import os

from multiprocessing import Pool

DATA_PATH = "dataset/"

def load_chunk(filename):
    if not os.path.exists(filename):
        return None
    try:
        with gzip.open(filename) as f:
            obs = pickle.load(f)
            N, A, H, W, C = obs.shape
            obs = obs.reshape(N * A, H, W, C)
            print(".", end='', flush=True)
            return obs
    except:
        return None

# load the data...
def load_dataset():
    print("Loading dataset:", end='')

    start_time = time.time()

    rollout_filenames = []
    for i in range(1000):
        fn = os.path.join(DATA_PATH, f"rollout_{i}.dat")
        if os.path.exists(fn):
            rollout_filenames.append(fn)
        else:
            break

    pool = Pool(24)
    data = pool.map(load_chunk, rollout_filenames)
    pool.close()
    pool.join()

    data = np.concatenate(data, axis=0)
    print()
    print(f"Loaded {len(data) / 1e6:.1f}M observations with shape {data.shape[1:]} in {time.time() - start_time:.1f}s")

    return data

if __name__ == "__main__":
    data = load_dataset()
    print("saving...")
    with open('dataset.dat', "wb") as f:
        pickle.dump(data, f, protocol=4) # required for large files
    print('done')
