#!/usr/bin/env python3
"""Contains the 'add_matrices' function"""


def add_matrices(mat1, mat2):
    """Adds two matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if isinstance(mat1, list) and isinstance(mat2, list):
        return [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]
    else:
        return mat1 + mat2


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
    mat1 = [1, 2, 3]
    mat2 = [4, 5, 6]
    print(add_matrices(mat1, mat2))
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices(mat1, mat2))
    mat1 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14, 15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]],
            [[[25, 26, 27, 28], [29, 30, 31, 32]],
             [[33, 34, 35, 36], [37, 38, 39, 40]],
             [[41, 42, 43, 44], [45, 46, 47, 48]]]]
    mat2 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[19, 110, 111, 112], [113, 114, 115, 116]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]],
            [[[125, 126, 127, 128], [129, 130, 131, 132]],
             [[133, 134, 135, 136], [137, 138, 139, 140]],
             [[141, 142, 143, 144], [145, 146, 147, 148]]]]
    mat3 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]],
            [[[125, 126, 127, 128], [129, 130, 131, 132]],
             [[141, 142, 143, 144], [145, 146, 147, 148]]]]
    print(add_matrices(mat1, mat2))
    print(add_matrices(mat1, mat3))
