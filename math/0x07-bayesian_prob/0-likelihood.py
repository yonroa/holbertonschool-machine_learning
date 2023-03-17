#!/usr/bin/env python3
"""This module cotains the function 'likelihood'"""

import numpy as np


def likelihood(x, n, P):
    """calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: numpy.ndarray containing the various hypothetical probabilities
            of developing severe side effects
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for number in P:
        if number < 0 or number > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    factorial = np.math.factorial(n) / (np.math.factorial(x)
                                        * np.math.factorial(n - x))
    likelihood = factorial * (P ** x) * ((1 - P) ** (n - x))

    return likelihood
