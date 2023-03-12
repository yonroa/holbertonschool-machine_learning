#!/usr/bin/env python3
"""This module contains the function 'minor'"""


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


def calc_minor(matrix, i, j):
    """
    Computes the minor of the element at row i and column j of a given matrix.
    """
    submatrix = []
    for row in range(len(matrix)):
        if row != i:
            subrow = []
            for col in range(len(matrix[row])):
                if col != j:
                    subrow.append(matrix[row][col])
            submatrix.append(subrow)

    # Calculate the determinant of the submatrix
    determinant = 0
    if len(submatrix) == 0:
        return 1
    if len(submatrix) == 2:
        determinant = submatrix[0][0] * submatrix[1][1] - \
            submatrix[0][1] * submatrix[1][0]
    else:
        for j in range(len(submatrix)):
            sign = (-1) ** j
            cofactor = calc_minor(submatrix, 0, j)
            determinant += sign * submatrix[0][j] * cofactor
    return determinant


def minor(matrix):
    """
    Computes the minor matrix of a given matrix.
    """
    check_for_matrix(matrix)
    check_for_square_matrix(matrix)
    minor_mat = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            # Compute the minor of the element at row i and column j
            submatrix = []
            for row in range(len(matrix)):
                if row != i:
                    subrow = []
                    for col in range(len(matrix[row])):
                        if col != j:
                            subrow.append(matrix[row][col])
                    submatrix.append(subrow)

            # Calculate the determinant of the submatrix
            determinant = 0
            if len(submatrix) == 0:
                return [[1]]
            if len(submatrix) == 2:
                determinant = submatrix[0][0] * submatrix[1][1]
                - submatrix[0][1] * submatrix[1][0]
            else:
                for k in range(len(submatrix)):
                    sign = (-1) ** k
                    cofactor = calc_minor(submatrix, 0, k)
                    determinant += sign * submatrix[0][k] * cofactor

            minor_row.append(determinant)

        minor_mat.append(minor_row)

    return minor_mat
