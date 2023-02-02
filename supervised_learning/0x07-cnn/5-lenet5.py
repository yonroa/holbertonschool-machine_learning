#!/usr/bin/env python3
"""Contains the function 'lenet5'"""

import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using keras

    Args:
        x: Keras containing the input images for the network
    """
    init = K.initializers.HeNormal()
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)
    flat = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(units=120,
                         kernel_initializer=init,
                         activation='relu')(flat)
    fc2 = K.layers.Dense(units=84,
                         kernel_initializer=init,
                         activation='relu')(fc1)
    fc3 = K.layers.Dense(units=10,
                         kernel_initializer=init,
                         activation='softmax')(fc2)
    model = K.Model(inputs=X, outputs=fc3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
