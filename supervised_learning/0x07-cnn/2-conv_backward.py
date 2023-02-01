#!/usr/bin/env python3
"""Contains the function 'conv_backward'"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer
    of a neural network

    Args:
        dZ: Array containing the partial derivatives with respect
            to the unactivated output of the convolutional layer
        A_prev: Array containing the output of the previous layer
        W: Array containing the kernels for the convolution
        b: Array containing the biases applied to the convolution
        padding: string that is either same or valid, indicating
            the type of padding used
        stride: Tuple containing the strides for the convolution
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape
    dW = np.zeros_like(W)
    da = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'valid':
        p_h, p_w = 0, 0
    elif padding == 'same':
        p_h = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        p_w = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1

    A_prev = np.pad(da, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                    mode='constant', constant_values=0)
    dA = np.pad(da, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                mode='constant', constant_values=0)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for h in range(c_new):
                    kernel = W[:, :, :, h]
                    dz = dZ[i, j, k, h]
                    calc = A_prev[i, sh * j: sh *
                                  j + kh, sw * k: sw * k + kw, :]
                    dW[:, :, :, h] += calc * dz
                    dA[i, sh * j: sh * j + kh, sw * k: sw *
                        k + kw, :] += np.multiply(kernel, dz)

    if padding == 'same':
        dA = dA[:, p_h: -p_h, p_w: -p_w, :]
    return dA, dW, db
