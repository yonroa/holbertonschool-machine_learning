#!/usr/bin/env python3
"""Contains the 'poly_integral' function"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if isinstance(poly, list) and all(
            isinstance(x, int) for x in poly) and isinstance(C, int):
        if len(poly) == 0:
            return [C]
        integrate = []
        integrate.append(C)
        for i in range(len(poly)):
            exp = i + 1
            coef = poly[i] / exp
            if coef.is_integer():
                integrate.append(int(coef))
            else:
                integrate.append(coef)
        return integrate
    return None


if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
