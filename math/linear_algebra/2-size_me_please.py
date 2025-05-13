#!/usr/bin/env python3
"""Contains the 'matrix_shape' function"""

shape = []


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape.clear()
    len_matrix(matrix)
    shape.reverse()
    return shape


def len_matrix(col):
    """Calculates the size of a vector"""
    if type(col) is list and len(col) > 1:
        len_matrix(col[0])
    add_shape(col)


def add_shape(vector):
    """Adds the size of the vector to shape array"""
    if type(vector) is not int:
        shape.append(len(vector))
