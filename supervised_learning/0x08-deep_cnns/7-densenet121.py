#!/usr/bin/env python3
"""Contains the function 'densenet121'"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks

    Args:
        growth_rate: Growth rate
        compression: Compression factor
    """
    init = K.initializers.HeNormal()
    layer = K.Input(shape=(224, 224, 3))
    filters = growth_rate * 2
    norm1 = K.layers.BatchNormalization(axis=3)(layer)
    activation1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=filters,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=init)(activation1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv1)
    X1, filters = dense_block(pool1, filters, growth_rate, layers=6)
    layer1, nb_filters = transition_layer(X1, nb_filters, compression)

    X2, filters = dense_block(layer1, filters, growth_rate, 12)
    layer2, nb_filters = transition_layer(X2, nb_filters, compression)

    X3, filters = dense_block(layer2, filters, growth_rate, 24)
    layer3, nb_filters = transition_layer(X3, filters, compression)

    X4, filters = dense_block(layer3, filters, growth_rate, 16)
    pool2 = K.layers.AveragePooling2D(pool_size=7)(X4)
    fc1 = K.layers.Dense(units=1000, activation='softmax',
                         kernel_initializer=init)(pool2)

    model = K.models.Model(inputs=layer, outputs=fc1)
    return model
