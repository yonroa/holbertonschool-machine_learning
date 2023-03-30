#!/usr/bin/env python3
"""This module contains the function 'viterbi'"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden
    states for a hidden markov model
    """
    try:
        if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
            return None, None
        if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
            return None, None
        if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
            return None, None
        if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
            return None, None

        T = Observation.shape[0]
        N, _ = Emission.shape

        if Transition.shape[0] != N or Transition.shape[1] != N:
            return None, None

        if Initial.shape[0] != N or Initial.shape[1] != 1:
            return None, None

        V = np.zeros((N, T))
        B = np.zeros((N, T))
        V[:, 0] = Initial.T * Emission[:, Observation[0]]
        B[:, 0] = 0

        for t in range(1, T):
            for n in range(N):
                V[n, t] = np.max(V[:, t - 1] * Transition[:, n]
                                 * Emission[n, Observation[t]])
                B[n, t] = np.argmax(
                    V[:, t - 1] * Transition[:, n]
                    * Emission[n, Observation[t]])

        path = [np.argmax(V[:, T - 1])]
        for i in range(T - 1, 0, -1):
            path.append(int(B[path[-1], i]))
        path = path[::-1]

        P = np.max(V[:, T - 1])

        return path, P
    except Exception:
        return None, None
