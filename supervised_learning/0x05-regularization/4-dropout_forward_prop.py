#!/usr/bin/env python3
"""Contain the function 'dropout_forward_prop'"""

import numpy as np


def softmax(z):
    """Softmax activation"""
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout

    Args:
        X: contains the input data for the network
        weights: dictionary of the weights and biases of the neural network
        L: number of layers in the network
        keep_prob: probability that a node will be kept
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        inputs = cache['A' + str(i)]
        z = np.dot(weights['W' + str(i + 1)], inputs) + \
            weights['b' + str(i + 1)]
        if i == L - 1:
            cache['A' + str(i+1)] = softmax(z)
        else:
            cache['D' + str(i+1)] = np.random.binomial(n=1,
                                                       p=keep_prob,
                                                       size=z.shape)
        fact = np.tanh(z) * cache['D' + str(i+1)]
        cache['A' + str(i+1)] = fact / keep_prob
        return cache
