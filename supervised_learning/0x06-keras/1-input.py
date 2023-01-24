#!/usr/bin/env python3
"""Contain the function 'build_model'"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes
            in each layer of the network
        activations: list containing the activation functions
            used for each layer of the network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    """
    reg = K.regularizers.L2(l2=lambtha)
    for i in range(len(layers)):
        if i == 0:
            inputs = K.Input(shape=(nx, ))
            pred = K.layers.Dense(units=layers[i],
                                  activation=activations[i],
                                  kernel_regularizer=reg)(inputs)
        else:
            pred = K.layers.Dense(units=layers[i],
                                  activation=activations[i],
                                  kernel_regularizer=reg)(pred)
        if i < len(layers) - 1:
            pred = K.layers.Dropout(rate=(1 - keep_prob))(pred)
    return K.Model(inputs=inputs, outputs=pred)
