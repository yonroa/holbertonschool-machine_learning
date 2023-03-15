#!/usr/bin/env python3
"""This module contains the function 'pca'"""

import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    covariance_matrix = np.cov(centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    transformation_matrix = eigenvectors[:, :ndim]
    transformed_data = np.dot(centered_data, transformation_matrix)
    return transformed_data
