#!/usr/bin/env python3
"""Contains the function 'transition_layer'"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely
    Connected Convolutional Networks

    Args:
        X: Output from the previous layer
        nb_filters: Integer representing the number of filters in X
        compression: Compression factor for the transition layer
    """
    init = K.initializers.HeNormal()
    norm1 = K.layers.BatchNormalization(axis=3)(X)
    activation1 = K.layers.Activation('relu')(norm1)
    filters = int(nb_filters * compression)
    layer = K.layers.Conv2D(filters=filters,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init)(activation1)
    pool = K.layers.AveragePooling2D(pool_size=2,
                                     strides=2)(layer)
    return pool, filters
