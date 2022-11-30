#!/usr/bin/env python3
"""Contains the 'add_matrices2D' function"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    sum_matrix = []
    for i in range(len(mat1)):
        row = []
        sum_matrix.append(row)
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
    print(sum_matrix)
    return sum_matrix


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
