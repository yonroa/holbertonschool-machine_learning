#!/usr/bin/env python3
"""This module contains the function 'baum_welch'"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model"""
    try:
        if not isinstance(Observations, np.ndarray) or len(
                Observations.shape) != 1:
            return None, None
        if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
            return None, None
        if not isinstance(Transition, np.ndarray) or len(
                Transition.shape) != 2:
            return None, None
        if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
            return None, None

        T = Observations.shape[0]
        M, N = Emission.shape

        if Transition.shape[0] != M or Transition.shape[1] != M:
            return None, None

        if Initial.shape[0] != M or Initial.shape[1] != 1:
            return None, None

        for i in range(iterations):
            alpha, _ = forward(Observations, Emission, Transition, Initial)
            beta, _ = backward(Observations, Emission, Transition, Initial)
            xi, gamma = expectation(
                Observations, Emission, Transition, Initial, alpha, beta)
            Transition, Emission = maximization(
                Observations, Transition, Emission, Initial, xi, gamma)

        return Transition, Emission
    except Exception:
        return None, None


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model"""
    try:
        N = Transition.shape[0]

        T = Observation.shape[0]

        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.sum(Transitions * F[:, t - 1]
                                 * Emissions)

        P = np.sum(F[:, -1])
        return P, F
    except Exception:
        None, None


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""
    try:
        T = Observation.shape[0]
        N, M = Emission.shape
        beta = np.zeros((N, T))
        beta[:, T - 1] = np.ones((N))

        for t in range(T - 2, -1, -1):
            for n in range(N):
                Transitions = Transition[n, :]
                Emissions = Emission[:, Observation[t + 1]]
                beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
        return P, beta
    except Exception:
        return None, None


def expectation(Observations, Emission, Transition, Initial, alpha, beta):
    """"""
    try:
        T = Observations.shape[0]
        M, N = Emission.shape

        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = alpha[i, t] * Transition[i, j] * \
                        Emission[j, Observations[t + 1]] * \
                        beta[j, t + 1]

            xi[:, :, t] /= np.sum(xi[:, :, t])

        for t in range(T):
            for i in range(M):
                gamma[i, t] = np.sum(xi[i, :, t])

        return xi, gamma
    except Exception:
        return None, None


def maximization(Observations, Transition, Emission, Initial, xi, gamma):
    """"""
    T = Observations.shape[0]
    M, N = Emission.shape

    for i in range(M):
        for j in range(M):
            Transition[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :T - 1])

    for i in range(M):
        for j in range(N):
            Emission[i, j] = np.sum(
                gamma[i, Observations == j]) / np.sum(gamma[i, :])

    return Transition, Emission
