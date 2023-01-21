#!/usr/bin/env python3
"""Contain the function 'create_confusion_matrix'"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix

    Args:
        labels: One-hot array containing the correct labels for each data point
        logits: One-hot array containing the predicted labels
    """
    result = np.zeros((len(labels[0]), len(labels[0])))
    labeln = np.where(labels == 1)[1]
    logitn = np.where(logits == 1)[1]
    for i in range(len(labeln)):
        result[labeln[i]][logits[i]] += 1
    return result
