#!/usr/bin/env python3
"""Contains the class 'DeepNeuralNetwork'"""

import numpy as np


def sigmoid(z):
    """Sigmoid or Logistic Activation Function"""
    return 1/(1 + np.exp(-z))


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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X: contains the input data
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            w = self.__weights["W{}".format(i + 1)]
            cache = self.__cache["A{}".format(i)]
            b = self.__weights["b{}".format(i + 1)]
            A = sigmoid(np.matmul(w, cache) + b)
            self.__cache["A{}".format(i + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example
        """
        loss = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return (-loss.sum() / len(Y.T))

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
        """
        A, self.__cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
        
        Args:
            Y: contains the correct labels for the input data
            cache: contains all the intermediary values of the network
            alpha: learning rate
        """
        m = len(Y[0])
        Dz = cache["A{}".format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A = cache["A{}".format(i - 1)]
            Dw = np.matmul(Dz, A.T) / m
            Db = np.sum(Dz)
            W = self.__weights["W{}".format(i)]
            Dz = np.matmul(W.T, Dz) * (A * (1 - A))
            self.__weights["W{}".format(i)] = self.__weights["W{}".format(i)] \
                - (alpha * Dw)
            self.__weights["b{}".format(i)] = self.__weights["b{}".format(i)] \
                - (alpha * Db)
