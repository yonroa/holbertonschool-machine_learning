#!/usr/bin/env python3
"""Contains the function 'one_hot_decode'"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels

    Args:
        one_hot: is a one-hot encoded numpy.ndarray
    Returns a numpy.ndarray, or None on failure
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
