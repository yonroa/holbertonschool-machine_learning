#!/usr/bin/env python3
"""This module contains the function 'expectation'"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM"""
    if ((type(X) is not np.ndarray or type(pi) is not np.ndarray
         or type(m) is not np.ndarray or type(S) is not np.ndarray
         or X.ndim != 2 or pi.ndim != 1 or m.ndim != 2 or S.ndim != 3
         or pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]
         or X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]
         or S.shape[1] != S.shape[2] or np.any(np.linalg.det(S) == 0)
         or not np.isclose(pi.sum(), 1))):
        return None, None

    pdfs = np.ndarray((m.shape[0], X.shape[0]))

    for cluster in range(m.shape[0]):
        pdfs[cluster] = pdf(X, m[cluster], S[cluster])

    pdfs = pdfs * pi[:, np.newaxis]
    sum = pdfs.sum(axis=0)
    expect = pdfs / sum

    return expect, np.log(sum).sum()
