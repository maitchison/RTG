# Just a script to run the pre-experiments.

import os

# default hyperparameters are fine, so just adjust the run settings

cmd = " ".join(["python train.py",
    "--train_scenarios=full",
    "--eval_scenarios=[]",
    "--micro_batch_size=1024",  # 1024 is slower, but can run two searches in parallel per GPU
    "--prediction_mode=others",
    "--save_model=recent",
    "--epochs=100",
    "--device='cuda:0'",
    "--deception_bonus=0.5",

])

print()
print(cmd)
print()
os.system(cmd)