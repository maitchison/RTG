from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input

import tensorflow as tf

import numpy as np

def cnn_default(scaled_images, **kwargs):
    """
    My modified CNN model,
    It's a bit slow but it gets the job done

    2080TI ~3,200 FPS
    CPU ~500 FPS

    Conv 3x3x32
    Conv 3x3x64
    MaxPool 2x2
    Conv 3x3x64
    MaxPool 2x2
    FC 64
    FC 64
    """

    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = tf.layers.max_pooling2d(layer_2, (2, 2), (2, 2))

    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = tf.layers.max_pooling2d(layer_3, (2, 2), (2, 2))
    layer_3_flat = conv_to_fc(layer_3)

    #print(f"model created with final dims={layer_3.shape} and flat_dim={layer_3_flat.shape[-1]}")

    layer_hidden = linear(layer_3_flat, "fc0", n_hidden=64, init_scale=np.sqrt(2))
    return activ(linear(layer_hidden, 'fc1', n_hidden=64, init_scale=np.sqrt(2)))

def cnn_fast(scaled_images, **kwargs):
    """
    CNN model optimized for speed
    It's a bit slow but it gets the job done (and much faster on CPU / low end GPU)

    2080TI ~4,000 FPS
    CPU ~2,000 FPS


    Conv 3x3x32 (stride 3)
    Conv 3x3x64
    Conv 3x3x64
    MaxPool 2x2
    FC 64
    FC 64
    """

    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=3, init_scale=np.sqrt(2), **kwargs))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, pad="SAME", init_scale=np.sqrt(2), **kwargs))

    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, pad="SAME", init_scale=np.sqrt(2), **kwargs))
    layer_3 = tf.layers.max_pooling2d(layer_3, (2, 2), (2, 2))
    layer_3_flat = conv_to_fc(layer_3)

    #print(f"model created with final dims={layer_3.shape} and flat_dim={layer_3_flat.shape[-1]}")

    layer_hidden = linear(layer_3_flat, "fc0", n_hidden=64, init_scale=np.sqrt(2))
    return activ(linear(layer_hidden, 'fc1', n_hidden=64, init_scale=np.sqrt(2)))
