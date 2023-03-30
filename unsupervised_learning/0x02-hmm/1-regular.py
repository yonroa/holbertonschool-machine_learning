#!/usr/bin/env python3
"""This module contains the function 'regular'"""

import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""
    try:
        if not isinstance(P, np.ndarray) or len(P.shape) != 2:
            return None

        if P.shape[0] != P.shape[1]:
            return None

        cols = P.shape[0]
        ans = np.ones((1, cols))
        eq = np.vstack([P.T - np.identity(cols), ans])
        results = np.vstack([np.zeros((cols, 1)), np.array([1])])

        statetionary = np.linalg.solve(eq.T.dot(eq), eq.T.dot(results)).T

        if len(np.argwhere(statetionary < 0)) > 0:
            return None

        return statetionary
    except Exception:
        return None
