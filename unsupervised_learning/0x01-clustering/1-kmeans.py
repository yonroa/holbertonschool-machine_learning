#!/usr/bin/env python3
"""This module contains the function 'kmeans'"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    C = np.random.uniform(min, max, (k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, np.newaxis] - C, axis=-1), axis=1)
        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = np.random.uniform(min, max, (1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)
    return C, clss
