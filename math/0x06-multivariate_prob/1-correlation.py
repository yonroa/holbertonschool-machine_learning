#!/usr/bin/env python3
"""This module contains the function 'correlation'"""

import numpy as np


def correlation(C):
    """calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    depends = np.diag(C)
    depends_dimensions_increased = np.expand_dims(depends, axis=0)
    standard_x = np.sqrt(depends_dimensions_increased)
    standard_product = np.dot(standard_x.T, standard_x)
    correlation = C / standard_product
    return correlation
