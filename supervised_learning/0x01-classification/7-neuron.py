#!/usr/bin/env python3
"""This module contain the class 'Neuron'"""

import numpy as np
import matplotlib.pyplot as plt


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
        self.__A = sigmoid(np.matmul(self.__W, X) + self.__b)
        return self.__A

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
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            A: contains the activated output of the neuron for each example
            alpha: the learning rate
        """
        m = X.shape[1]
        Dz = A - Y
        Dw = (1 / m) * np.matmul(Dz, X.T)
        Db = (1 / m) * np.sum(Dz)
        self.__W = self.__W - (alpha * Dw)
        self.__b = self.__b - (alpha * Db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neuron

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
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            cost = self.cost(Y, self.__A)
            if verbose and i % step == 0:
                costs.append(cost)
                iteration.append(i)
                print(f"Cost after {i} iterations: {cost}")
        if graph:
            plt.plot(iteration, costs, "b")
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show

        return self.evaluate(X, Y)
