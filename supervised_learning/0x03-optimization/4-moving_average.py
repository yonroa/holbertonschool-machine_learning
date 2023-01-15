#!/usr/bin/env python3
"""Contains the function 'moving_average'"""

import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average
    """
    w_avgs = []
    avg = 0
    for x, n in enumerate(data):
        avg = ((beta*avg) + ((1-beta)*n))
        bias_correct = avg / (1-(beta**(x+1)))
        w_avgs.append(bias_correct)
    return w_avgs
