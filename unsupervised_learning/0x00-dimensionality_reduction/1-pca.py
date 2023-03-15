#!/usr/bin/env python3
"""This module contains the function 'pca'"""

import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    x_mean = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(x_mean)
    W = vh.T[:, :ndim]
    T = np.matmul(x_mean, W)
    return T
