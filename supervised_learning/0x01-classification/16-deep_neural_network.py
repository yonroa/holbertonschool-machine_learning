#!/usr/bin/env python3
"""Contains the class 'DeepNeuralNetwork'"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize the class 'DeepNeuralNetwork'

        Args:
            nx: number of input features
            layers: list representing the number of nodes
                    in each layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or list == []:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            if type(layers[i]) != int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            b = np.zeros((layers[i], 1))
            w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            self.weights["b{}".format(layers[i] + 1)] = b
            self.weights["W{}".format(layers[i] + 1)] = w
            nx = layers[i]
