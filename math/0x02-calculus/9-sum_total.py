#!/usr/bin/env python3
"""Contains the 'summation_i_squared' function"""


def summation_i_squared(n):
    """Calculates the summatory of i^2"""
    if type(n) != int or n < 0:
        return None
    return (n * (n + 1) * ((2 * n) + 1)) // 6


if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))
