#!/usr/bin/env python3
"""This module contains the function 'expectation_maximization'"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, l_ = expectation(X, pi, m, S)

    for i in range(iterations):
        g, l_ = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)
        g, l_new = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i,
                l_.round(5)
            ))
        if abs(l_new - l_) <= tol:
            break
        l_ = l_new
    g, l_ = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i + 1,
            l_.round(5)
        ))
    return pi, m, S, g, l_
