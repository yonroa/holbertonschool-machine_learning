#!/usr/bin/env python3
"""Contains the 'Binomial' class"""


def factorial(num):
    """Find the factorial of 'num'"""
    fact = 1
    for i in range(1, num + 1):
        fact *= i
    return fact


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Instantiation of the 'Binomial' class.

        Args:
            data: list of the data to be used to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a “success”
        """
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            summatory = 0
            for x in data:
                summatory += (x - (sum(data) / len(data))) ** 2
            var = (1 / len(data)) * summatory
            p = 1 - (var / (sum(data) / len(data)))
            self.n = round((sum(data) / len(data) / p))
            self.p = (sum(data) / len(data)) / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”

        Args:
            k: number of “successes”
        """
        try:
            k = int(k)
        except Exception:
            return 0
        if k < 0:
            return 0
        coef = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        return coef * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”

        Args:
            k: number of “successes”
        """
        try:
            k = int(k)
        except Exception:
            return 0
        if k < 0:
            return 0
        summatory = 0
        for i in range(k + 1):
            coef = factorial(self.n) / (factorial(i) * factorial(self.n - i))
            summatory += coef * (self.p ** i) * ((1 - self.p) ** (self.n - 1))
        return summatory
