#!/usr/bin/env python3
"""Contains the 'add_matrices2D' function"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""
    for vector_1 in mat1:
        for vector_2 in mat2:
            if len(vector_1) != len(vector_2):
                return None
    new_matrix = []
    for x in range(0, len(mat1)):
        new_vector = []
        for y in range(0, len(mat1[0])):
            new_vector.append(mat1[x][y] + mat2[x][y])
        new_matrix.append(new_vector)
    return new_matrix
