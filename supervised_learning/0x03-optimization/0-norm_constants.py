#!/usr/bin/env python3
"""Contains the function 'normalization_constants'"""

import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix

    Args:
        X: Is the array to normalize
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
