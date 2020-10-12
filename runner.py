# just a way to save runs in a script
import os

def run():
    os.system(
        """python train.py train """ +
        """--device=2 --run="V2.6 r2g2" """ +
        """--train_scenarios=r2g2 """ +
        """--eval_scenarios=r2g2 """ +
        """--parallel_envs=128 """     # needs to be a multiple of minibatches, which is 4
    )

if __name__ == "__main__":
    run()