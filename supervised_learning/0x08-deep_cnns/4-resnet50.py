#!/usr/bin/env python3
"""Contains the function 'resnet50'"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in Deep
    Residual Learning for Image Recognition (2015)
    """
    init = K.initializers.HeNormal()
    layer = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            kernel_initializer=init,
                            strides=(2, 2),
                            padding='same')(input)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation('relu')(norm1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(activation1)

    filters = (64, 64, 256)
    projection1 = projection_block(pool1, filters, s=1)
    identity1 = identity_block(projection1, filters)
    identity2 = identity_block(identity1, filters)

    filters = (128, 128, 512)
    projection2 = projection_block(identity2, filters)
    identity3 = identity_block(projection2, filters)
    identity4 = identity_block(identity3, filters)
    identity5 = identity_block(identity4, filters)

    filters = (256, 256, 1024)
    projection3 = projection_block(identity5, filters)
    identity6 = identity_block(projection3, filters)
    identity7 = identity_block(identity6, filters)
    identity8 = identity_block(identity7, filters)
    identity9 = identity_block(identity8, filters)
    identity10 = identity_block(identity9, filters)

    filters = (512, 512, 2048)
    projection4 = projection_block(identity10, filters)
    identity11 = identity_block(projection4, filters)
    identity12 = identity_block(identity11, filters)

    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(identity12)
    fc1 = K.layers.Dense(units=1000,
                         activation='softmax',
                         kernel_initializer=init)(pool2)

    model = K.models.Model(inputs=layer, outputs=fc1)
    return model
