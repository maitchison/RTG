# just a way to save runs in a script
import os

def run():
    # os.system(
    #     """python train.py train """ +
    #     """--device=2 --run="V2.7.1 r2g4" """ +
    #     """--algo_params="{'nminibatches':12}" """+       # this should usually be number of players * 2 (to fit into 11GB)
    #     """--train_scenarios=r2g4 """ +
    #     """--eval_scenarios="[ """ +
    #
    #     """ ['r2g4', None, None, None], """ +
    #     """ ['r2g3', None, None, None], """ +
    #     """ ['r2g2', None, None, None], """ +
    #
    #     """ ['r2g4', 'wander', None, None], """ +
    #     """ ['r2g4', None, 'wander', None], """ +
    #     """ ['r2g4', 'rush_general', None, None], """ +
    #     """ ['r2g4', 'rush_general_cheat', None, None], """ +
    #
    #     """ ['r2g2', 'rush_general', None, None], """ +
    #     """ ['r2g3', 'rush_general', None, None], """ +
    #
    #     """ ]" """ +
    #     """--parallel_envs=128 """     # needs to be a multiple of minibatches, which is usually 4
    # )


    os.system(
        """python train.py train """ +
        """--device=2 --run="V2.7.1 full" """ +
        """--algo_params="{'nminibatches':48}" """+       # this should usually be number of players * 2 (to fit into 11GB)
        """--train_scenarios=full """ +
        """--parallel_envs=128"""     # needs to be a multiple of minibatches, which is usually 4
    )

if __name__ == "__main__":
    run()