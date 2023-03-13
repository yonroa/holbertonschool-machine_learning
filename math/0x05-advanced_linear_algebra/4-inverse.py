#!/usr/bin/env python3
"""This module contains the function 'inverse'"""


def inverse(matrix):
    """calculates the inverse of a matrix"""
    check_for_matrix(matrix)
    check_for_square_matrix(matrix)
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square")
    determinant_matrix = determinant(matrix)
    if determinant_matrix == 0:
        return None
    if len(matrix) == 1:
        return [[1 / matrix[0][0]]]
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = [[cofactor_matrix[j][i]
                        for j in range(len(matrix))]
                       for i in range(len(matrix))]
    inverse_matrix = [[adjugate_matrix[i][j] /
                       determinant_matrix for j in range(len(matrix))]
                      for i in range(len(matrix))]
    return inverse_matrix


def adjugate(matrix):
    """calculates the adjugate matrix of a matrix"""
    if len(matrix) == 1:
        return [[1]]
    cofactor_matrix = cofactor(matrix)
    adjugate_m = [[cofactor_matrix[j][i]
                   for j in range(len(matrix))] for i in range(len(matrix))]
    return adjugate_m


def cofactor(matrix):
    """calculates the cofactor matrix of a matrix"""
    cofactor_matrix = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix)):
            minor_matrix = [row[:j] + row[j+1:]
                            for row in (matrix[:i]+matrix[i+1:])]
            sign = (-1)**(i+j)
            cofactor_row.append(sign * determinant(minor_matrix))
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix


def check_for_matrix(matrix):
    """Checks if the argument is a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for vector in matrix:
        if not isinstance(vector, list):
            raise TypeError("matrix must be a list of lists")


def check_for_square_matrix(matrix):
    """Checks if the argument is a square matrix"""
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    for vector in matrix:
        if len(matrix) != len(vector):
            raise ValueError("matrix must be a non-empty square matrix")


def determinant(matrix):
    """calculates the determinant of a matrix"""
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
