#!/usr/bin/env python3
"""This module contains the function 'absorbing'"""

import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    try:
        if not isinstance(P, np.ndarray):
            return False

        if len(P.shape) != 2:
            return False

        if P.shape[0] != P.shape[1]:
            return False

        for elem in np.sum(P, axis=1):
            if not np.isclose(elem, 1):
                return False

        diagonal = np.diag(P)

        if (diagonal == 1).all():
            return True

        absorb = (diagonal == 1)

        for row in range(len(diagonal)):
            for col in range(len(diagonal)):
                if P[row, col] > 0 and absorb[col]:
                    absorb[row] = 1

        if (absorb == 1).all():
            return True

        return False
    except Exception:
        return False
