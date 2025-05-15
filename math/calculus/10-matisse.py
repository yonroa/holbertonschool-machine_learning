#!/usr/bin/env python3
"""Contains the 'poly_derivative' function"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if isinstance(poly, list) and all(
            isinstance(x, int) for x in poly) and len(poly) > 0:
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
    poly = [0, 3, -2, 1, 0]
    print(poly_derivative(poly))
