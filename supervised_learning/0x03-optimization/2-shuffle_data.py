#!/usr/bin/env python3
"""Contains the function 'shuffle_data'"""

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way

    Args:
        X: Is first array to shuffle
        Y: Is the second array to shuffle
    """
    shuffle = np.random.permutation(len(X))
    return X[shuffle], Y[shuffle]
