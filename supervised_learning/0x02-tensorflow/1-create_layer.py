#!/usr/bin/env python3
"""Contains the function 'create_layer'"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Returns the tensor output of the layer"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel=init,
        name=layer
    )
    return layer(prev)
