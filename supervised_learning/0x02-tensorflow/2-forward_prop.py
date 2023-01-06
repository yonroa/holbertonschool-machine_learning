#!/usr/bin/env python3
"""Contains the function 'forward_prop'"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[], index=0):
    """Creates the forward propagation graph for the neural network"""
    if index >= len(layer_sizes):
        return x
    layer = create_layer(x, layer_sizes[index], activations[index])
    return forward_prop(layer, layer_sizes, activations, index + 1)
