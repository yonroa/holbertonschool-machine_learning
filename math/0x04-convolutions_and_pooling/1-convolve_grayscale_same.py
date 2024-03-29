#!/usr/bin/env python3
"""Contains the 'convolve_grayscale_same' function"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images

    Args:
        images: Array containing multiple grayscale images
        kernel: Array containing the kernel for the convolution
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = int((kh - 1) / 2)
    pw = int((kw - 1) / 2)
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convolve = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            convolve[:, i, j] = np.sum(
                images[:, i: i + kh, j: j + kw] * kernel, axis=(1, 2))
    return convolve
