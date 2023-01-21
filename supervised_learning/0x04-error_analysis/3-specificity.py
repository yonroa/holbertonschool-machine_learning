#!/usr/bin/env python3
"""Contain the function 'specificity'"""

import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: where row indices represent the correct
            labels and column indices represent the predicted labels
    """
    collapsed = np.sum(confusion.flatten())
    especify = np.zeros((confusion.shape[0]))
    for i in range(confusion.shape[0]):
        negative = collapsed - \
            (np.sum(confusion[i]) + np.sum(confusion[:, i]) - confusion[i][i])
        especify[i] = negative / \
            (negative + (np.sum(confusion[:, i]) - confusion[i][i]))
    return especify
