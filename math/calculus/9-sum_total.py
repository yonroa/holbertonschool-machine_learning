#!/usr/bin/env python3
"""Contains the 'summation_i_squared' function"""


def summation_i_squared(n):
    """Calculates the summatory of i^2"""
    if isinstance(n, int) and n > 0:
        return int((n / 6) * (n + 1) * (2 * n + 1))
    return None


if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))
