"""
A multi-agent version of PPO

"""

from stable_baselines import PPO2

class MARL_Environment_Adapter():
    """
    Very simple idea: Treat all other agents as part of the environmennt.

    The advantage is we can now plug in any RL algorithms and use it in MARL setting

    This makes the environment non-stationary, so convergence is definitely not a guarintee, and in practice
    will probably not work in many cases.
    """

    def __init__(self, policy, env, base_algorithm, **kwargs):
        """
        :param policy:
        :param env:
        :param base_algorithm:
        :param kwargs: any additonal args are passed to PPO agents.
        """

        #


