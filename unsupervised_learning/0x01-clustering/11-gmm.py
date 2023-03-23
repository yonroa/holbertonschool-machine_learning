#!/usr/bin/env python3
"""This module contains the function 'BIC'"""

import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset"""
    if len(X.shape) != 2:
        return None, None, None, None, None

    if type(k) != int or k <= 0:
        return None, None, None, None, None

    gm = sklearn.mixture.GaussianMixture(n_components=k)
    gm.fit(X)

    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_

    clss = gm.predict(X)
    bic = gm.bic(X)

    return pi, m, S, clss, bic
