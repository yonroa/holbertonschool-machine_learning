#!/usr/bin/env python3
"""This module contains the class 'MultiNormal'"""

import numpy as np


class MultiNormal:
    """represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initialize the class MultiNormal"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")
        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True).reshape(d, 1)
        self.cov = np.matmul(data - self.mean, (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != \
                self.cov.shape[0]:
            raise ValueError(
                "x must have the shape ({}, 1)".format(self.cov.shape[0]))
        x_mean = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)
        pdf = (1 / np.sqrt(((2 * np.pi) ** self.cov.shape[0]) * cov_det)) * \
            np.exp(-0.5 * np.matmul(np.matmul(x_mean.T, cov_inv), x_mean))
        return pdf[0][0]
