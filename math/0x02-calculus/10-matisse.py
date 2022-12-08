#!/usr/bin/env python3
"""Contains the 'poly_derivative' function"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if isinstance(poly, list) and all(
            isinstance(x, (int, float)) for x in poly):
        derivate = []
        for i in range(len(poly)):
            exp = i - 1
            coef = poly[i] * i
            if exp >= 0:
                derivate.append(coef)
        if len(derivate) == 0:
            return [0]
        return derivate
    return None


if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_derivative(poly))
