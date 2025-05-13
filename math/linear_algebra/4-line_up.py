#!/usr/bin/env python3
"""Contains the 'add_arrays' function"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    sum_array = []
    for x in range(0, len(arr1)):
        sum_array.append(arr1[x] + arr2[x])
    return sum_array
