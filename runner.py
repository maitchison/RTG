# just a way to save runs in a script
import os

def run_v2_3_r2g4():
    os.system(
        """python train.py train """+
        """--device=1 --run="V2.4 r2g4" """+
        """--train_scenarios=r2g4 """+
        """--eval_scenarios=r2g4 """+
        """--vary_team_player_counts=True """+
        """--parallel_envs=4 """     # needs to be a multiple of minibatches, which is 4
    )

if __name__ == "__main__":
    run_v2_3_r2g4()