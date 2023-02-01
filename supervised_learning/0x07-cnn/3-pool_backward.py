#!/usr/bin/env python3
"""Contains the function 'pool_backward'"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network

    Args:
        dA: Array containing the partial derivatives with respect
            to the output of the pooling layer
        A_prev: Array containing the output of the previous layer
        kernel_shape: Tuple containing the size of the kernel for the pooling
        stride: Tuple containing the strides for the convolution
        mode: string containing either max or avg, indicating whether
            to perform maximum or average pooling
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    pool = np.zeros_like(A_prev)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for h in range(c):
                    if mode == 'avg':
                        avg = dA[i, j, k, h] / kh / kw
                        pool[i, sh * j: sh * j + kh, sw * k: sw *
                             k + kw, h] += (np.ones((kh, kw)) * avg)
                    if mode == 'max':
                        box = A_prev[i, sh * j:sh * j +
                                     kh, sw * k: sw * k + kw, h]
                        mask = (box == np.max(box))
                        pool[i, sh * j:sh * j + kh, sw * k: sw *
                             k + kw, h] += (mask * dA[i, j, k, h])

    return pool
