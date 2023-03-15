#!/usr/bin/env python3
"""This module contains the function 'pca'"""

import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    u, s, vh = np.linalg.svd(X)
    W = vh.T[:, :ndim]
    T = np.matmul(X, W)
    return T
