#!/usr/bin/env python3
"""Contains the 'convolve' function"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels

    Args:
        images: Array containing multiple images
        kernels: Array containing the kernels for the convolution
        padding: Tuple of (ph, pw) or 'same' or 'valid'
        stride: Tuple of (sh, sw)
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride
    ph, pw = (0, 0)

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    if isinstance(padding, tuple):
        ph, pw = padding

    convolve_h = int(((h + (2 * ph) - kh) / sh) + 1)
    convolve_w = int(((w + (2 * pw) - kw) / sw) + 1)

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    convolve = np.zeros((m, convolve_h, convolve_w, nc))
    for i in range(convolve_h):
        for j in range(convolve_w):
            for k in range(nc):
                convolve[:, i, j, k] = np.sum(
                    images[:, i * sh: i * sh + kh,
                           j * sw: j * sw + kw] * kernels[:, :, :, k],
                    axis=(1, 2, 3))
    return convolve
