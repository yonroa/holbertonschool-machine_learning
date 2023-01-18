#!/usr/bin/env python3
"""Contains the 'pool' function"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images

    Args:
        images: Array containing multiple images
        kernel_shape: Tuple containing the kernel shape for the pooling
        padding: Tuple of (ph, pw) or 'same' or 'valid'
        stride: Tuple of (sh, sw)
        mode: indicates the type of pooling
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = int(((h - kh) / sh) + 1)
    pw = int(((w - kw) / sw) + 1)
    pool = np.zeros((m, ph, pw, c))

    for i in range(ph):
        for j in range(pw):
            if mode == 'max':
                pool[:, i, j, :] = np.max(
                    images[:, i * sh: i * sh + kh,
                           j * sw: j * sw + kw], axis=(1, 2))
            if mode == 'avg':
                pool[:, i, j, :] = np.average(
                    images[:, i * sh: i * sh + kh,
                           j * sw: j * sw + kw], axis=(1, 2))
    return pool
