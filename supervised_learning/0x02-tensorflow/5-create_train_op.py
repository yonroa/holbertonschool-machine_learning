#!/usr/bin/env python3
"""Contains the function 'create_train_op'"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    return tf.train.GradientDescentOptimizer(learning_rate=alpha,
                                             ).minimize(loss)
