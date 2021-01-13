# just a file to kepp track of parameters for main experiments


"""

# r2g2
python train.py train --train_scenario="r2g2" --eval_scenarios="[]" --prediction_mode='action' --verbose=2 --device="cuda:0" --deception_bonus='[0,0,0]' --run="exp_r2g2_db0"
python train.py train --train_scenario="r2g2" --eval_scenarios="[]" --prediction_mode='action' --verbose=2 --device="cuda:1" --deception_bonus='[3,0,0]' --run="exp_r2g2_db3"


# wolf game
python train.py train --train_scenario="wolf_sheep_nv" --eval_scenarios="[]" --prediction_mode='action' --verbose=2 --device="cuda:2" --deception_bonus='[0,0,0]' --run="exp_wolf_db0"
python train.py train --train_scenario="wolf_sheep_nv" --eval_scenarios="[]" --prediction_mode='action' --verbose=2 --device="cuda:3" --deception_bonus='[3,0,0]' --run="exp_wolf_db3"

# full game
python train.py train --train_scenario="5p" --eval_scenarios="[]" --prediction_mode='action' --verbose=2 --device="cuda:0" --deception_bonus='[0,0,0]' --run="exp_full_db0"
python train.py train --train_scenario="5p" --eval_scenarios="[]" --prediction_mode='action' --verbose=2 --device="cuda:1" --deception_bonus='[3,0,0]' --run="exp_full_db3"

"""
