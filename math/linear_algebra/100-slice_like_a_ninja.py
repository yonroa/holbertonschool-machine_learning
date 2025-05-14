#!/usr/bin/env python3
"""Contains the 'np_slice' function"""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    new_matrix = matrix[:]
    slices = [slice(None)] * matrix.ndim
    for axis, tup in axes.items():
        slices[axis] = slice(*tup)
    return new_matrix[tuple(slices)]


if __name__ == "__main__":
    import numpy as np
    mat1 = np.array([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10]])
    print(np_slice(mat1, axes={1: (1, 3)}))
    mat2 = np.array([[[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10]],
                     [[11, 12, 13, 14, 15],
                      [16, 17, 18, 19, 20]],
                     [[21, 22, 23, 24, 25],
                      [26, 27, 28, 29, 30]]])
    print(np_slice(mat2, axes={0: (2, ), 2: (None, None, -2)}))
    print(mat2)
