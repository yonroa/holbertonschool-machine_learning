#!/usr/bin/env python3
"""This module contains the function 'determinant'"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for vector in matrix:
        if not isinstance(vector, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for j in range(n):
            minor = [[matrix[i][k]
                      for k in range(n) if k != j] for i in range(1, n)]
            det += matrix[0][j] * (-1) ** j * determinant(minor)
        return det
