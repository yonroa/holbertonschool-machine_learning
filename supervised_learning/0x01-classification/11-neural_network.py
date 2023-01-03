#!/usr/bin/env python3
"""This module contain the class 'NeuralNetwork'"""

import numpy as np


def sigmoid(z):
    """Sigmoid or Logistic Activation Function"""
    return 1/(1 + np.exp(-z))


class NeuralNetwork:
    """defines a neural network with one hidden
    layer performing binary classification
    """

    def __init__(self, nx, nodes):
        """Initialize the class 'NeuralNetwork'

        Args:
            nx: number of input features
            nodes: number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros([nodes, 1])
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter fuction for the weights vector of the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter fuction for the bias of the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter fuction for the activated output of the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter fuction for the weights vector of the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter fuction for the bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter fuction for the activated output
        for the output neuron (prediction)
        """
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X: contains the input data
        """
        self.__A1 = sigmoid(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = sigmoid(np.matmul(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        
        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example
        """
        loss = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return (-loss.sum() / len(Y.T))
