#!/usr/bin/env python3
"""Contains the class 'DeepNeuralNetwork'"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(z):
    """Sigmoid or Logistic Activation Function"""
    return 1/(1 + np.exp(-z))

def tanh(x):
    """Tanh function"""
    num = np.exp(x) - np.exp(-x)
    den = (np.exp(x) + np.exp(-x))
    return num / den


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Initialize the class 'DeepNeuralNetwork'

        Args:
            nx: number of input features
            layers: list representing the number of nodes
                    in each layer of the network
            activation: represents the type of activation
                    function used in the hidden layers
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
        self.__activation = activation
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
    def activation(self):
        """Type of activation function to use"""
        return self.__activation

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
            b = self.__weights["b{}".format(i + 1)]
            if self.__activation == 'sig':
                activation = sigmoid(np.dot(w, X) + b)
            elif self.__activation == 'tanh':
                activation = tanh(np.dot(w, X) + b)
            X = activation
            self.__cache['A' + str(i + 1)] = activation
        return self.__cache['A{}'.format(self.__L)], self.__cache

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
        self.__cache = cache
        m = len(Y[0])
        for i in range(self.__L, 0, -1):
            if self.__activation == 'sig':
                if i == self.__L:
                    dzl = self.__cache['A' + str(i)] - Y
            if self.__activation == 'tanh':
                if i == self.__L:
                    num = (self.__cache['A' + str(i)] - Y)
                    deno = self.__cache['A' + str(i)]
                    factor = num / deno
                    dzl = factor * (1 + self.__cache['A' + str(i)])
            X = self.__cache['A' + str(i - 1)]
            weight_derivative_l = np.dot(X, dzl.T) / m
            bias_derivative_l = np.sum(dzl, axis=1, keepdims=True) / m
            if self.__activation == 'sig':
                valor = self.__cache['A' + str(i - 1)]
                d_activation = valor * (1 - valor)
            elif self.__activation == 'tanh':
                valor = self.__cache['A' + str(i - 1)]
                d_activation = 1 - (tanh(valor)) ** 2
            dzl_1 = np.dot(self.__weights['W' + str(i)].T, dzl) * d_activation
            dzl = dzl_1
            wl = self.__weights['b' + str(i)]
            restw = (alpha * bias_derivative_l)
            bl = self.__weights['W' + str(i)]
            restb = (alpha * weight_derivative_l.T)
            self.__weights['b' + str(i)] = wl - restw
            self.__weights['W' + str(i)] = bl - restb

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, A)
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

    def save(self, filename):
        """Saves the instance object to a file in pickle format

        Args:
            filename: file to which the object should be saved
        """
        if type(filename) is not str:
            return None
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        try:
            with open(filename, "wb") as f:
                obj = pickle.dump(self, f)
                return obj
        except Exception:
            return None

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object

        Args:
            filename: file from which the object should be loaded
        """
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
                return obj
        except Exception:
            return None
