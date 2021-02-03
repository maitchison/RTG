# This script will generate the results used in the ICML paper and is provided for reproducibility.
# Time to run is approximately 4 days if run on 4 GPUs, however this script will run experiments sequentially on one GPU
# On a multi-gpu machine it is recommended to execute each run in parallel and to modifying the --device='cuda:0'
#   appropriately.
# Note: A RTX2080 can comfortably run 2 experiments simultaneously with little performance impact.

import os

os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413a_db00' --prediction='action' --deception_bonus='(0,0,0)' --device='cuda:0' --epochs=300 --seed=0")
os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413b_db00' --prediction='action' --deception_bonus='(0,0,0)' --device='cuda:0' --epochs=300 --seed=1")
os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413c_db00' --prediction='action' --deception_bonus='(0,0,0)' --device='cuda:0' --epochs=300 --seed=2")
os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413d_db00' --prediction='action' --deception_bonus='(0,0,0)' --device='cuda:0' --epochs=300 --seed=3")

os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413a_db05' --prediction='action' --deception_bonus='(0.5,0,0)' --device='cuda:0' --epochs=300 --seed=4")
os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413b_db05' --prediction='action' --deception_bonus='(0.5,0,0)' --device='cuda:0' --epochs=300 --seed=5")
os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413c_db05' --prediction='action' --deception_bonus='(0.5,0,0)' --device='cuda:0' --epochs=300 --seed=6")
os.system("python train.py train --train_scenario='rescue_training' --eval_scenarios='[]' --micro=1024 --verbose=2 --run='rescue413d_db05' --prediction='action' --deception_bonus='(0.5,0,0)' --device='cuda:0' --epochs=300 --seed=7")

# this will test agents against a mixture over all agents.
os.system("python arena.py")