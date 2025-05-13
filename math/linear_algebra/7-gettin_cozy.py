#!/usr/bin/env python3
"""Contains the 'cat_matrices2D' function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return [*mat1, *mat2]
    elif axis == 1 and len(mat1) == len(mat2):
        return list(map(lambda arr1, arr2: cat_arrays(arr1, arr2), mat1, mat2))
    else:
        return None


def cat_arrays(arr1, arr2):
    """Concatenates two arrays"""
    new_array = arr1.copy()
    for x in range(0, len(arr2)):
        new_array.append(arr2[x])
    return new_array
