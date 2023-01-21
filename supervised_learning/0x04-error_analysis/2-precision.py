#!/usr/bin/env python3
"""Contain the function 'precision'"""

import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix

    Args:
        confusion: where row indices represent the correct
            labels and column indices represent the predicted labels
    """
    prec = np.zeros((confusion.shape[0]))
    for i in range(confusion.shape[0]):
        prec[i] = confusion[i][i] / np.sum(confusion[:, i])
    return prec
