#!/usr/bin/env python3
"""Contains the 'convolve_grayscale_valid' function"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images

    Args:
        images: Array containing multiple grayscale images
        kernel: Array containing the kernel for the convolution
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    convolve = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            convolve[:, i, j] = np.sum(
                images[:, i: i + kh, j: j + kw] * kernel, axis=(1, 2))
    return convolve
