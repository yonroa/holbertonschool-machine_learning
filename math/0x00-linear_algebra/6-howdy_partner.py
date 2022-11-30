#!/usr/bin/env python3
"""Contains the 'cat_arrays' function"""


def cat_arrays(arr1, arr2):
    """Concatenates two arrays"""
    new_array = arr1.copy()
    new_array.extend(arr2)
    return new_array


if __name__ == "__main__":
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]
    print(cat_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
