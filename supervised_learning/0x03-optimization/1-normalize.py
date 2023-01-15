#!/usr/bin/env python3
"""Contains the function 'normalize'"""

import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes) a matrix

    Args:
        X: Is the array to normalize
        m: Array that contains the mean of all features of X
        s: Array  that contains the standard deviation of all features of X
    """
    return (X - m) / s
