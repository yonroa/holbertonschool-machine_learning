#!/usr/bin/env python3
"""Contains the 'matrix_transpose' function"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    transpose = []
    columns = len(matrix)
    rows = len(matrix[0])
    for row in range(0, rows):
        new_vector = []
        for col in range(0, columns):
            new_vector.append(matrix[col][row])
        transpose.append(new_vector)
    return transpose