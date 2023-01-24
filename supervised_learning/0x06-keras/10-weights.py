#!/usr/bin/env python3
"""Contain the functions 'save_weights' and 'load_weights'"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves a model's weights

    Args:
        network: model whose weights should be saved
        filename: path of the file that the weights should be saved to
        save_format:  format in which the weights should be saved
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads a model's weights

    Args:
        network: model to which the weights should be loaded
        filename: path of the file that the model should be loaded from
    """
    return network.load_weights(filepath=filename)
