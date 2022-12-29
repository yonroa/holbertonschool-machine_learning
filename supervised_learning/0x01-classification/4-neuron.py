#!/usr/bin/env python3
"""This module contain the class 'Neuron'"""

import numpy as np


def sigmoid(z):
    """Sigmoid or Logistic Activation Function"""
    return 1/(1 + np.exp(-z))


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
        self.__W = np.random.normal(size=(1, nx), scale=1.0)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector of the neuron"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias of the neuron"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output of the neuron"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X: contains the input data
        """
        self.__A = sigmoid(np.dot(self.W, X) + self.b)
        return self.A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: contains the activated output of the neuron for each example
        """
        loss = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return (-loss.sum() / len(Y.T))

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
        """
        self.forward_prop(X)
        return (np.rint(self.A).astype(int), self.cost(Y, self.A))
