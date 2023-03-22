#!/usr/bin/env python3
"""This module contains the function 'variance'"""

import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None
    centroids = C[:, np.newaxis]
    distances = np.sqrt(((X - centroids)**2).sum(axis=2))
    min_distances = np.min(distances, axis=0)
    var = np.sum(min_distances**2)
    return var
