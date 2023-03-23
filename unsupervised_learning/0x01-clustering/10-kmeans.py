#!/usr/bin/env python3
"""This module contains the function 'kmeans'"""

import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    if len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
