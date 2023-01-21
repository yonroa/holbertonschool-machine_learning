#!/usr/bin/env python3
"""Contain the function 'l2_reg_cost'"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of the weights and biases
        L: number of layers in the neural network
        m: number of data points used
    """
    norma = 0
    for key, value in weights.items():
        if key[0] == 'W':
            norma += np.linalg.norm(value)
    ridge_regression = (lambtha / (2 * m)) * norma
    return (cost + ridge_regression)
