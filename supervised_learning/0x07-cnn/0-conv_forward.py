#!/usr/bin/env python3
"""Contains the function 'conv_forward'"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional
    layer of a neural network

    Args:
        A_prev: Array containing the output of the previous layer
        W: Array containing the kernels for the convolution
        b: Array containing the biases applied to the convolution
        activation: activation function applied to the convolution
        padding: string that is either same or valid, indicating
            the type of padding used
        stride: Tuple containing the strides for the convolution
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        p_h = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        p_w = ((((w_prev - 1) * sw) + kw - w_prev) // 2)
    elif padding == 'valid':
        p_h, p_w = 0, 0
    elif isinstance(padding, tuple):
        p_h, p_w = padding

    images = np.pad(A_prev, ((0, 0), (p_h, p_h),
                    (p_w, p_w), (0, 0)), mode='constant', constant_values=0)
    h_out = ((h_prev - kh + 2 * p_h) // sh) + 1
    w_out = ((w_prev - kw + 2 * p_w) // sw) + 1
    conv = np.zeros((m, h_out, w_out, c_new))

    for i in range(c_new):
        for j in range(h_out):
            for k in range(w_out):
                p_images = np.sum(np.multiply(
                    images[:, sh * j: sh * j + kh, sw * k: sw * k + kw],
                    W[:, :, :, i]), axis=(1, 2, 3))
                conv[:, j, k, i] = activation((p_images + b[:, :, :, i]))

    return conv
