#!/usr/bin/env python3
"""Contain the function 'l2_reg_cost'"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost of the network
        without L2 regularization
    """
    return (cost + tf.losses.get_regularization_losses())
