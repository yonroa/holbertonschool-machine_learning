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
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        negative = list(filter(lambda x: x <= 0, layers))
        if len(negative) > 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            def first():
                if i == 0:
                    factor1, factor2 = np.random.randn(
                        layers[i], nx), np.sqrt(2 / nx)
                    self.__weights['W' + str(i + 1)] = factor1 * factor2
                    return
                factor1, factor2 = np.random.randn(
                    layers[i], layers[i - 1]), np.sqrt(2 / layers[i - 1])
                self.__weights['W' + str(i + 1)] = factor1 * factor2
            first()
            zeros = np.zeros(layers[i])
            self.__weights['b' + str(i + 1)] = zeros.reshape(layers[i], 1)

    @property
    def L(self):
        """Getter for the number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """Getter for a dictionary to hold all
        intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """Getter for A dictionary to hold all
        weights and biased of the network
        """
        return self.__weights
