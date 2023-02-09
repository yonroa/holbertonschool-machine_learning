#!/usr/bin/env python3
"""Contains the function 'projection_block'"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in Deep
    Residual Learning for Image Recognition (2015)

    Args:
        A_prev: Output from the previous layer
        filters: Tuple or list containing convolutions
        s: stride of the first convolution in both the
            main path and the shortcut connection
    """
    init = K.initializers.HeNormal()
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11,
                            strides=s,
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

    conv4 = K.layers.Conv2D(filters=F12,
                            strides=s,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init)(A_prev)
    norm4 = K.layers.BatchNormalization(axis=3)(conv4)

    add_layer = K.layers.Add()([norm3, norm4])
    activation3 = K.layers.Activation('relu')(add_layer)
    return activation3
