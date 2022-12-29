#!/usr/bin/env python3
"""This module contain the class 'Neuron'"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize the class 'Neuron'

        Args:
            nx: number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(size=(1, nx), scale=1.0)
        self.b = 0
        self.A = 0
