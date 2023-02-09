#!/usr/bin/env python3
"""Contains the function 'identity_block'"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in Deep
    Residual Learning for Image Recognition (2015)

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing filters
    """
    init = K.initializers.HeNormal()
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation('relu')(norm1)

    conv2 = K.layers.Conv2D(filters=F3,
                            padding='same',
                            kernel_size=(3, 3),
                            kernel_initializer=init)(activation1)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    activation2 = K.layers.Activation('relu')(norm2)

    conv3 = K.layers.Conv2D(filters=F12,
                            padding='same',
                            kernel_size=(1, 1),
                            kernel_initializer=init)(activation2)
    norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    add_layer = K.layers.Add()([norm3, A_prev])
    activation3 = K.layers.Activation('relu')(add_layer)
    return activation3
