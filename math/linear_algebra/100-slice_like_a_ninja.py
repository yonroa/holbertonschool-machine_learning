#!/usr/bin/env python3
"""Contains the 'np_slice' function"""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    for axis, slice in axes.items():
        if len(slice) == 1:
            