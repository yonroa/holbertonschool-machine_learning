#!/usr/bin/env python3
"""Contains 'line' function"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot 'y' as a line graph"""

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r')
    plt.axis([0, 10, -50, 1050])
    plt.show()


if __name__ == "__main__":
    line()
