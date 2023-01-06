#!/usr/bin/env python3
"""Contains the function 'create_placeholders'"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, name="x", shape=(None, nx))
    y = tf.placeholder(tf.float32,  name="y", shape=(None, classes))
    return x, y
