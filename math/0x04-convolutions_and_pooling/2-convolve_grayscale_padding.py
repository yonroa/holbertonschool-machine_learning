#!/usr/bin/env python3
"""Contains the 'convolve_grayscale_padding' function"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding

    Args:
        images: Array containing multiple grayscale images
        kernel: Array containing the kernel for the convolution
        padding: Tuple of (ph, pw)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    convolve_h = h + (2 * ph) - kh + 1
    convolve_w = w + (2 * pw) - kw + 1
    img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convolve = np.zeros((m, convolve_h, convolve_w))
    for i in range(convolve_h):
        for j in range(convolve_w):
            convolve[:, i, j] = np.sum(
                img[:, i: i + kh, j: j + kw] * kernel, axis=(1, 2))
    return convolve
