#!/usr/bin/env python3
"""Contains the function 'one_hot_encode'"""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix

    Args:
        Y: contains numeric class labels
        classes: maximum number of classes found in Y
    Return a one-hot encoding of Y with shape (classes, m),
    or None on failure
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.eye(classes)[Y]
        return one_hot.T
    except Exception:
        return None
