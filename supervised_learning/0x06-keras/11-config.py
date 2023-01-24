#!/usr/bin/env python3
"""Contain the functions 'save_config' and 'load_config'"""

import tensorflow.keras as K


def save_config(network, filename):
    """saves a model's configuration in JSON format

    Args:
        network: model whose configuration should be saved
        filename: path of the file that the configuration should be saved to
    """
    model = network.to_json()
    with open(filename, 'w') as file:
        file.write(model)
    return None


def load_config(filename):
    """loads a model with a specific configuration

    Args:
        filename: path of the file containing the model's
        configuration in JSON format
    """
    with open(filename, 'r') as file:
        model = file.read()
    return K.models.model_from_json(model)
