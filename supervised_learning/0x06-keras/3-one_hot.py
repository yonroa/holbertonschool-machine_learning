#!/usr/bin/env python3
"""Contain the function 'one_hot'"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix

    Args:
        The last dimension of the one-hot matrix must
        be the number of classes
    """
    return K.utils.to_categorical(labels, num_classes=classes)
