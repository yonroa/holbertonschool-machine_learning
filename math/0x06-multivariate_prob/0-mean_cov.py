#!/usr/bin/env python3
"""This module contains the function 'mean_cov'"""

import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    d = X.shape[1]
    mean = np.mean(X, axis=0).reshape(1, d)
    points = (X.shape[0] - 1)
    X_mean = X - mean
    cov = np.matmul(X_mean.T, X_mean) / points
    return (mean, cov)
