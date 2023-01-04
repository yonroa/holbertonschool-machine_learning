#!/usr/bin/env python3
"""This module contain the class 'NeuralNetwork'"""

import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
        """
        self.__A1, self.__A2 = self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            A1: output of the hidden layer
            A2: predicted output
            alpha: learning rate
        """
        m = len(Y[0])
        Dz2 = A2 - Y
        Dw2 = (1 / m) * np.matmul(A1, Dz2.T)
        Db2 = (1 / m) * np.sum(Dz2)
        Dz1 = np.matmul(self.__W2.T, Dz2) * (A1 * (1 - A1))
        Dw1 = (1 / m) * np.matmul(X, Dz1.T)
        Db1 = (1 / m) * np.sum(Dz1)

        self.__W1 = self.__W1 - (alpha * Dw1.T)
        self.__b1 = self.__b1 - (alpha * Db1)
        self.__W2 = self.__W2 - (alpha * Dw2.T)
        self.__b2 = self.__b2 - (alpha * Db2)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neural network

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: boolean that defines whether or not to print
                    information about the training
            graph: boolean that defines whether or not to graph information
                    about the training once the training has completed
        """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iteration = []
        for i in range(iterations + 1):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            cost = self.cost(Y, self.__A2)
            if verbose and i % step == 0:
                costs.append(cost)
                iteration.append(i)
                print(f"Cost after {i} iterations: {cost}")
        if graph:
            plt.plot(iteration, costs, "b")
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)
