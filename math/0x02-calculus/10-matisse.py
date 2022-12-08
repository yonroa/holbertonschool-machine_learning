#!/usr/bin/env python3
"""Contains the 'poly_derivative' function"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if isinstance(poly, list):
        derivate = []
        for i in range(len(poly)):
            exp = i - 1
            coef = poly[i] * i
            if exp >= 0:
                derivate.append(coef)
        return derivate
    else:
        return None


if __name__ == "__main__":
    poly = [1, -5, 0, 3, 1]
    print(poly_derivative(poly))
