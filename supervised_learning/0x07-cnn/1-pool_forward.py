#!/usr/bin/env python3
"""Contains the function 'pool_forward'"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network

    Args:
        A_prev: Array containing the output of the previous layer
        kernel_shape: Tuple containing the size of the kernel for the pooling
        stride: Tuple containing the strides for the convolution
        mode: string containing either max or avg, indicating whether
            to perform maximum or average pooling
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    if mode == 'max':
        op = np.max
    elif mode == 'avg':
        op = np.mean

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    pool = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            pool[:, i, j, :] = op(
                A_prev[:, sh * i: sh * i + kh,
                       sw * j: sw * j + kw], axis=(1, 2))

    return pool
