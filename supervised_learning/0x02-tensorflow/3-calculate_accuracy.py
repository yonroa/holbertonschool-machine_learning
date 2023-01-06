#!/usr/bin/env python3
"""Contains the function 'calculate_accuracy'"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
    return accuracy
