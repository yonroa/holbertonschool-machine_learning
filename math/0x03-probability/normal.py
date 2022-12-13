#!/usr/bin/env python3
"""Contains the 'Normal' class"""


def erf(x):
    """Calculates error function or erf"""
    pi = 3.1415926536
    pol = ((x) - ((x ** 3)/3) + ((x**5)/10) - ((x**7)/42) + ((x**9)/216))
    return (2 / (pi ** 0.5)) * pol


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Instantiation of the 'Exponential' class.

        Args:
            data: list of the data to be used to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            sumatoria = 0
            for x in data:
                sumatoria += (x - self.mean) ** 2
            self.stddev = (sumatoria / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value

        Args:
            x: x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-score of a given z-value

        Args:
            z: z-value
        """
        return (self.stddev * z) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value

        Args:
            x: x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        term1 = 1 / (self.stddev * ((2 * pi) ** 0.5))
        term2 = e ** (-((x - self.mean) ** 2) / (2 * (self.stddev ** 2)))
        return term1 * term2

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value

        Args:
            x: x-value
        """
        term1 = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return (1 / 2) * (1 + erf(term1))
