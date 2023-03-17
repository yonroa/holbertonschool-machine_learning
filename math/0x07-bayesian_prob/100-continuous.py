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


def intersection(x, n, P, Pr):
    """calculates the intersection of obtaining this data with the various
    hypothetical probabilities

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: numpy.ndarray containing the various hypothetical probabilities
            of developing severe side effects
        Pr: numpy.ndarray containing the prior beliefs of P
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for number in P:
        if number < 0 or number > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for number in Pr:
        if number < 0 or number > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    intersection = likelihood(x, n, P) * Pr
    return intersection


def marginal(x, n, P, Pr):
    """calculates the marginal probability of obtaining the data

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: numpy.ndarray containing the various hypothetical probabilities
            of developing severe side effects
        Pr: numpy.ndarray containing the prior beliefs of P
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for number in P:
        if number < 0 or number > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for number in Pr:
        if number < 0 or number > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    marginal = np.sum(intersection(x, n, P, Pr))
    return marginal


def posterior(x, n, p1, p2):
    """calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        p1: lower bound on the range
        p2: upper bound on the range
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise TypeError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise TypeError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    P = np.linspace(0, 1, 101)
    Pr = np.ones(P.shape) / len(P)
    posterior = np.sum(intersection(x, n, P, Pr) * (P >= p1) * (P <= p2))
    return posterior
