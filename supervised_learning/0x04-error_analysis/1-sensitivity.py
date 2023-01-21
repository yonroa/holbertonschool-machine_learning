#!/usr/bin/env python3
"""Contain the function 'sensitivity'"""

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: where row indices represent the correct
            labels and column indices represent the predicted labels
    """
    sens = np.zeros((confusion.shape[0]))
    for i in range(confusion.shape[0]):
        sens[i] = confusion[i][i] / np.sum(confusion[i])
    return sens
