# just a way to save runs in a script
import os

def run():
    os.system(
        """python train.py train """ +
        """--device=1 --run="V2.7 r2g2" """ +
        """--train_scenarios=r2g2 """ +
        """--eval_scenarios="[ """ +
        """ ['r2g2', None, None, None], """ +
        """ ['r2g2', 'wander', None, None], """ +
        """ ['r2g2', None, 'wander', None], """ +
        """ ['r2g2', 'rush_general', None, None], """ +
        """ ['r2g2', 'rush_general_cheat', None, None] """ +
        """ ]" """ +
        """--parallel_envs=128 """     # needs to be a multiple of minibatches, which is usually 4
    )

if __name__ == "__main__":
    run()