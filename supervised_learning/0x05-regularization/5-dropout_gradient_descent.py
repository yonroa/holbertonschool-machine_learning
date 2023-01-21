#!/usr/bin/env python3
"""Contain the function 'dropout_gradient_descent'"""

import numpy as np


def dev_tanh(A):
    """Derivate of tanh"""
    return 1 - (A ** 2)


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Conducts forward propagation using Dropout

    Args:
        Y: One-hot array that contains the correct labels for the data
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each
            layer of the neural network
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
    """
    m = Y.shape[1]
    copy_weigths = weights.copy()
    for i in range(L, 0, -1):
        if i == L:
            layer_error = cache['A' + str(i)] - Y
        else:
            fact = np.dot(copy_weigths['W' + str(i + 1)].T, prev_error)
            layer_error = fact * dev_tanh(cache['A' + str(i)])
            layer_error *= cache['D' + str(i)] / keep_prob
        dev_cost = np.dot(layer_error, cache['A' + str(i - 1)].T) / m
        dev_cost_b = np.sum(layer_error, axis=1, keepdims=True) / m
        weights['W' + str(i)] = copy_weigths['W' + str(i)] - alpha * dev_cost
        weights['b' + str(i)] = copy_weigths['b' + str(i)] - alpha * dev_cost_b
        prev_error = layer_error
