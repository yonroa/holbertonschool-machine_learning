#!/usr/bin/env python3
"""This module contains the function 'BIC'"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters for a GMM using
    the Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None, None
    if type(kmin) != int or kmin <= 0:
        return None, None, None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) != int or kmax <= 0:
        return None, None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None, None

    l = []
    b = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, loglike = expectation_maximization(X, k, iterations,
                                                        tol, verbose)
        p = (k * m.shape[1]) + (k * m.shape[1] * (m.shape[1] + 1) / 2)
        bic = p * np.log(X.shape[0]) - 2 * loglike
        l.append(loglike)
        b.append(bic)

    l = np.asarray(l)
    b = np.asarray(b)
    best_k = np.argmin(b) + kmin
    pi, m, S, g, loglike = expectation_maximization(X, best_k, iterations,
                                                    tol, verbose)
    best_result = (pi, m, S)

    return best_k, best_result, l, b
