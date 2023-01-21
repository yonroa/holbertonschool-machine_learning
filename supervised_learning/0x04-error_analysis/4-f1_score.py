#!/usr/bin/env python3
"""Contain the function 'f1_score'"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix

    Args:
        confusion: where row indices represent the correct
            labels and column indices represent the predicted labels
    """
    pre, sens = precision(confusion), sensitivity(confusion)
    return 2*((pre * sens) / (pre + sens))
