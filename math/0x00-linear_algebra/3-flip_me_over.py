#!/usr/bin/env python3
"""Contains the 'matrix_transpose' function"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    transverse = []
    for i in range(len(matrix[0])):
        row = []
        transverse.append(row)
        for j in range(len(matrix)):
            row.append(matrix[j][i])
    return transverse
