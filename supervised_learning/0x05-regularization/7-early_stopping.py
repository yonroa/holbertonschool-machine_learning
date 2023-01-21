#!/usr/bin/env python3
"""Contain the function 'early_stopping'"""

import tensorflow.compat.v1 as tf


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early

    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met
    """
    cost_step = opt_cost - cost
    if cost_step > threshold:
        count = 0
    else:
        count += 1
    if count < patience:
        return False, count
    else:
        return True, count
