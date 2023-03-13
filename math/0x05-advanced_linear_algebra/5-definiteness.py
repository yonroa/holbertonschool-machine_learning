#!/usr/bin/env python3
"""This module contains the function 'definiteness'"""

import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix) == 0 or matrix.shape[0] != matrix.shape[1]:
        return None
    eigenvalues, _ = np.linalg.eig(matrix)
    if all(eigenvalues > 0):
        return "Positive definite"
    elif all(eigenvalues < 0):
        return "Negative definite"
    elif any(eigenvalues == 0):
        if all(eigenvalues >= 0):
            return "Positive semi-definite"
        elif all(eigenvalues <= 0):
            return "Negative semi-definite"
        else:
            return "Indefinite"
    else:
        if all(eigenvalues >= 0):
            return "Positive semi-definite"
        elif all(eigenvalues <= 0):
            return "Negative semi-definite"
        else:
            return "Indefinite"
