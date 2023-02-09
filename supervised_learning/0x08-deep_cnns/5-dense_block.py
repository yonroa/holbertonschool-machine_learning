#!/usr/bin/env python3
"""Contains the function 'dense_block'"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely
    Connected Convolutional Networks

    Args:
        X: Output from the previous layer
        nb_filters: Integer representing the number of filters in X
        growth_rate: Growth rate for the dense block
        layers: Number of layers in the dense block
    """
    init = K.initializers.HeNormal()

    for layer in range(layers):
        norm1 = K.layers.BatchNormalization(axis=3)(X)
        activation1 = K.layers.Activation('relu')(norm1)
        conv1 = K.layers.Conv2D(filters=(4 * growth_rate),
                                kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=init)(activation1)
        norm2 = K.layers.BatchNormalization(axis=3)(conv1)
        activation2 = K.layers.Activation('relu')(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer=init)(activation2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
