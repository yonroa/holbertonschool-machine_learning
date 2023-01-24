#!/usr/bin/env python3
"""Contain the function 'predict'"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes a prediction using a neural network

    Args:
        network: network model to make the prediction with
        data: input data to make the prediction with
        verbose: boolean that determines if output should
            be printed during the prediction process
    """
    return network.predict(x=data, verbose=verbose)
