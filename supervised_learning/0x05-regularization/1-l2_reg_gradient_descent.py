#!/usr/bin/env python3
"""Contain the function 'l2_reg_gradient_descent'"""
import numpy as np


def dev_tanh(A):
    """Derivate of tanh"""
    return 1 - (A ** 2)


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Args:
        Y: One-hot array that contains the correct labels for the data
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: learning rate
        lambtha: L2 regularization parameter
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
        dev_cost = np.dot(layer_error, cache['A' + str(i - 1)].T) / m
        reg = dev_cost + ((lambtha / m) * copy_weigths['W' + str(i)])
        dev_cost_b = np.sum(layer_error, axis=1, keepdims=True) / m
        weights['W' + str(i)] = copy_weigths['W' + str(i)] - alpha * reg
        weights['b' + str(i)] = copy_weigths['b' + str(i)] - alpha * dev_cost_b
        prev_error = layer_error
