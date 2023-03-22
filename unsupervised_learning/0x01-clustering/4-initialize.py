#!/usr/bin/env python3
"""This module contains the function 'initialize'"""

import numpy as np


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    kmeans = __import__('1-kmeans').kmeans
    centroids, clss = kmeans(X, k)
    pi = np.full((k,), 1 / k)
    m = centroids
    S = np.full((k, X.shape[1], X.shape[1]), np.identity(X.shape[1]))
    return pi, m, S
