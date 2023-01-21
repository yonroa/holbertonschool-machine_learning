#!/usr/bin/env python3
"""Contain the function 'dropout_create_layer'"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        keep_prob: probability that a node will be kept
    """
    initialice = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                       mode='fan_avg')
    reg = tf.keras.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_regularizer=reg,
                            kernel_initializer=initialice,
                            name='layer')
    return layer(inputs=prev)
