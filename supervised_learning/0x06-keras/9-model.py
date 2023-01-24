#!/usr/bin/env python3
"""Contain the functions 'save_model' and 'load_model'"""

import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model

    Args:
        network: the model to save
        filename: path of the file that the model should be saved to
    """
    K.models.save_model(model=network, filepath=filename)
    return None


def load_model(filename):
    """loads an entire model

    Args:
        filename: path of the file that the model should be loaded from
    """
    return K.models.load_model(filepath=filename)
