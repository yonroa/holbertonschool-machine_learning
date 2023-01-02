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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            A: contains the activated output of the neuron for each example
            alpha: the learning rate
        """
        m = Y.shape[1]
        Dz = A - Y
        Dw = (1 / m) * np.matmul(Dz, X.T)
        Db = (1 / m) * np.sum(Dz)
        self.__W = self.__W - (alpha * Dw)
        self.__b = self.__b - (alpha * Db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise TypeError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise TypeError("alpha must be positive")

        for _ in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)
