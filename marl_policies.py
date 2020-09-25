"""
This just multiples LSTM policy linear layer initialization by 0.01

Note: this probably isn't needed as in MARL the initialization doesn't matter so much, we need to be constantly
adapting so entropy bonus is probably a better way to go. Exploration really needs to be never-give-up style
rather than expore early on, and settle down (because env is dynamic).

"""

from new_models import cnn_default

import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import RecurrentActorCriticPolicy, nature_cnn

class CentralCritic():
    """
    Learns value for all players in a MARL setting from global observation
    :return:
    """

    def __init__(self, sess, observation_space, n_players):
        """                 
        :param sess: tf session 
        :param observation_space: input observational space
        :param n_players:
        :param n_batch:
        :return: 
        """

        # todo: implement grad clipping maybe?
        # todo: tf logs

        # setup inputs
        self.observations_placeholder = tf.placeholder(tf.uint8, [None, *observation_space])
        self.target_value_placeholder = tf.placeholder(tf.float32, [None])

        # create our model
        with tf.variable_scope("model"):
            processed_input = tf.cast(tf.float32, self.observations_placeholder) / 255.0
            self.value = linear(cnn_default(processed_input), "value", n_players)
            self.params = tf.trainable_variables()

        # setup train step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=2.5e-4, epsilon=1e-5)
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.value - self.target_value_placeholder))
        optimizer_operation = self.optimizer.minimize(self.loss, self.params)
        self._train = optimizer_operation.run()

        self.sess = sess

    def predict(self, observations):
        """
        Predicts values for each player given global obs
        :param observations: input observations to predict value from
        :return: tensor containing predicted values for each player
        """
        return self.sess.run([self.value], {self.observations_placeholder: observations})


    def train(self, global_obs, target_values):
        """
        trains model on given batch of global observation data

        :param global_obs: tensor of dims [n, *observation_space]
        :param target_values: tensor of dims [n]
        :return:
        """

        batch_size = len(global_obs)
        indexes = np.arange(batch_size)
        mb_size = 64

        losses = []

        for epoch_num in range(4):
            np.random.shuffle(indexes)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb_indexes = indexes[start:end]
                slices = [data[mb_indexes] for data in (global_obs, target_values)]

                mb_loss, _ = self.sess.run([self.loss, self._train], {
                    self.observations_placeholder: slices[0],
                    self.target_value_placeholder: slices[1]
                })

                losses.append(mb_loss)

        print("global vf loss", np.asarray(losses).mean())




