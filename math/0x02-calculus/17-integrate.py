#!/usr/bin/env python3
"""Contains the 'poly_integral' function"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if poly and isinstance(poly, list) and (
            isinstance(C, int) or isinstance(C, float)) and all(
            isinstance(x, (int, float)) for x in poly):
        if poly == [0]:
            return[C]
        integrate = []
        integrate.append(C)
        for i in range(len(poly)):
            coef = poly[i] / (i + 1)
            if coef.is_integer():
                integrate.append(int(coef))
            else:
                integrate.append(coef)
        return integrate
    return None


if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
