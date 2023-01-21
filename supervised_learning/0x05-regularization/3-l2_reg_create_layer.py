#!/usr/bin/env python3
"""Contain the function 'l2_reg_create_layer'"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        lambtha: L2 regularization parameter
    """
    reg = tf.keras.regularizers.L2(l2=lambtha)
    weight = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=weight,
                            kernel_regularizer=reg,
                            name='layer')
    return layer(inputs=prev)
