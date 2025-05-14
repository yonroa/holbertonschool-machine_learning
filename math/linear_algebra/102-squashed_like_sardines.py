#!/usr/bin/env python3
"""Contains the 'cat_matrices' function"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None
    if axis == 0:
        return mat1 + mat2
    return [cat_matrices(m1, m2, axis=axis - 1) for m1, m2 in zip(mat1, mat2)]


def matrix_shape(matrix):
    """Calcula la forma de una matriz"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break  # Evita errores si hay una lista vacía
        matrix = matrix[0]
    return shape


if __name__ == "__main__":
    import numpy as np
    import time

    mat1 = [1, 2, 3]
    mat2 = [4, 5, 6]
    np_mat1 = np.array(mat1)
    np_mat2 = np.array(mat2)

    t0 = time.time()
    m = cat_matrices(mat1, mat2)
    t1 = time.time()
    print(t1 - t0)
    print(m)
    t0 = time.time()
    np.concatenate((np_mat1, np_mat2))
    t1 = time.time()
    print(t1 - t0, "\n")

    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    np_mat1 = np.array(mat1)
    np_mat2 = np.array(mat2)

    t0 = time.time()
    m = cat_matrices(mat1, mat2)
    t1 = time.time()
    print(t1 - t0)
    print(m)
    t0 = time.time()
    np.concatenate((np_mat1, np_mat2))
    t1 = time.time()
    print(t1 - t0, "\n")

    t0 = time.time()
    m = cat_matrices(mat1, mat2, axis=1)
    t1 = time.time()
    print(t1 - t0)
    print(m)
    t0 = time.time()
    np.concatenate((mat1, mat2), axis=1)
    t1 = time.time()
    print(t1 - t0, "\n")

    mat3 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14, 15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]],
            [[[25, 26, 27, 28], [29, 30, 31, 32]],
             [[33, 34, 35, 36], [37, 38, 39, 40]],
             [[41, 42, 43, 44], [45, 46, 47, 48]]]]
    mat4 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
            [[19, 110, 111, 112], [113, 114, 115, 116]],
            [[117, 118, 119, 120], [121, 122, 123, 124]]],
            [[[125, 126, 127, 128], [129, 130, 131, 132]],
            [[133, 134, 135, 136], [137, 138, 139, 140]],
            [[141, 142, 143, 144], [145, 146, 147, 148]]]]
    mat5 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
            [[117, 118, 119, 120], [121, 122, 123, 124]]],
            [[[125, 126, 127, 128], [129, 130, 131, 132]],
            [[141, 142, 143, 144], [145, 146, 147, 148]]]]
    np_mat3 = np.array(mat3)
    np_mat4 = np.array(mat4)
    np_mat5 = np.array(mat5)

    t0 = time.time()
    m = cat_matrices(mat3, mat4, axis=3)
    t1 = time.time()
    print(t1 - t0)
    print(m)
    t0 = time.time()
    np.concatenate((np_mat3, np_mat4), axis=3)
    t1 = time.time()
    print(t1 - t0, "\n")

    t0 = time.time()
    m = cat_matrices(mat3, mat5, axis=1)
    t1 = time.time()
    print(t1 - t0)
    print(m)
    t0 = time.time()
    np.concatenate((np_mat3, np_mat5), axis=1)
    t1 = time.time()
    print(t1 - t0, "\n")

    m = cat_matrices(mat2, mat5)
    print(m)
