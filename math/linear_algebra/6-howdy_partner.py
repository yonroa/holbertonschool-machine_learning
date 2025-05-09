#!/usr/bin/env python3
"""Contains the 'cat_arrays' function"""


def cat_arrays(arr1, arr2):
    """Concatenates two arrays"""
    new_array = arr1.copy()
    for x in range(0, len(arr2)):
        new_array.append(arr2[x])
    return new_array