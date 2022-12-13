#!/usr/bin/env python3
"""Contains the 'Poisson' class"""


def factorial(num):
    """Find the factorial of 'num'"""
    fact = 1
    for i in range(1, num + 1):
        fact *= i
    return fact


class Poisson:
    """Represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Instantiation of the 'Poisson' class.

        Args:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”

        Args:
            k: number of “successes”
        """
        e = 2.7182818285
        try:
            k = int(k)
        except Exception:
            return 0
        return (e**-self.lambtha) * (self.lambtha**k) / factorial(k)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”

        Args:
            k: number of “successes”
        """
        e = 2.7182818285
        try:
            k = int(k)
        except Exception:
            return 0
        CDF = 0
        for i in range(k + 1):
            CDF += (e**-self.lambtha) * (self.lambtha**i) / factorial(i)
        return CDF
