#!/usr/bin/env python3
"""Contains the function 'inception_block'"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block as described in Going
    Deeper with Convolutions (2014)

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing filters
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.HeNormal()

    conv1 = K.layers.Conv2D(filters=F1,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=init)(A_prev)
    conv2 = K.layers.Conv2D(filters=F3R,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=init)(A_prev)
    conv3 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same',
                            kernel_initializer=init)(conv2)
    conv4 = K.layers.Conv2D(filters=F5R,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=init)(A_prev)
    conv5 = K.layers.Conv2D(filters=F5,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same',
                            kernel_initializer=init)(conv4)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                                  strides=(1, 1))(A_prev)
    conv6 = K.layers.Conv2D(filters=FPP,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=init)(pool1)

    concatenate = K.layers.concatenate(inputs=[conv1, conv3, conv5, conv6])
    return concatenate
