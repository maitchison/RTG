from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input
import tensorflow as tf

import numpy as np

def single_layer_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """

    # stub simplified model based on social influence as IM paper.
    # actually I changed it otherwise we get a very large linear layer that's too slow for my poor pc... :(

    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    #layer_1 = tf.layers.max_pooling2d(layer_1, (2,2), (2,2))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = tf.layers.max_pooling2d(layer_2, (2, 2), (2, 2))

    layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = tf.layers.max_pooling2d(layer_3, (2, 2), (2, 2))
    layer_3 = conv_to_fc(layer_3)

    print("model created with final dims ", layer_3.shape)

    layer_hidden = linear(layer_3, "fc0", n_hidden=64, init_scale=np.sqrt(2))
    return activ(linear(layer_hidden, 'fc1', n_hidden=64, init_scale=np.sqrt(2)))