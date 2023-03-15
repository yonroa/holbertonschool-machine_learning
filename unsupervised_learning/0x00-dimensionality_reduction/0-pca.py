#!/usr/bin/env python3
"""This module contains the function 'pca'"""

import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    u, s, vh = np.linalg.svd(X)
    c_var = np.cumsum(s) / np.sum(s)
    max_val = np.argmax(c_var >= var)
    W = vh.T[:, :max_val + 1]
    return W
