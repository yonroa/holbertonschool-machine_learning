#!/usr/bin/env python3
"""Contains the 'add_matrices2D' function"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""
    if len(mat1) == len(mat2):
        sum_matrix = []
        for i in range(len(mat1)):
            sum_matrix.append(add_arrays(mat1[i], mat2[i]))
        if None in sum_matrix:
            return None
        return sum_matrix
    return None


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    sum_array = []
    for i in range(len(arr1)):
        sum_array.append(arr1[i] + arr2[i])
    return sum_array


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, mat2))
    print(mat1)
    print(mat2)
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
