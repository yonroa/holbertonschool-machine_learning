#!/usr/bin/env python3
"""Contains the 'mat_mul' function"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    product = [[0 for x in range(len(mat2[0]))] for y in range(len(mat1))]
    for row in range(len(mat1)):
        for col in range(len(mat2[0])):
            for elt in range(len(mat2)):
                product[row][col] += (mat1[row][elt] * mat2[elt][col])
    return product


if __name__ == "__main__":
    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    print(mat_mul(mat1, mat2))
