#!/usr/bin/env python3
"""Contain the function 'test_model'"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network

    Args:
        network: network model to test
        data: input data to test the model with
        labels: correct one-hot labels of data
        verbose: boolean that determines if output
            should be printed during the testing process
    """
    return network.evaluate(x=data,
                            y=labels,
                            verbose=verbose)
