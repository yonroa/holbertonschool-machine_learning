#!/usr/bin/env python3
"""Contains the 'cat_matrices2D' function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return [*mat1, *mat2]
    elif axis == 1 and len(mat1) == len(mat2):
        return list(map(lambda arr1, arr2: cat_arrays(arr1, arr2), mat1, mat2))
    else:
        return None

def cat_arrays(arr1, arr2):
    """Concatenates two arrays"""
    new_array = arr1.copy()
    new_array.extend(arr2)
    return new_array

if __name__ == "__main__":
    m1 = [[51, 24, 73], [93, 45, 77]]
    m2 = [[], [], []]
    m = cat_matrices2D(m1, m2)
    print(m)
    m2 = [[51, 24, 73], [93, 45, 77]]
    m1 = [[], [], []]
    m = cat_matrices2D(m1, m2, axis=0)
    print(m)
    m1 = [[75, 23, 58], [32, 5, 67], [34, 65, 22]]
    m2 = [[], [], []]
    m = cat_matrices2D(m1, m2, axis=1)
    if type(m) is not list or m is m1 or m is m2 or not len(m) or type(m[0]) is not list:
        print("Not a new matrix")
    print(m)
