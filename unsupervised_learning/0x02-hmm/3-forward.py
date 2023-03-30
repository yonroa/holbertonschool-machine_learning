#!/usr/bin/env python3
"""This module contains the function 'forward'"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model"""
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

        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.sum(F[:, t - 1] * Transition[:, n]
                                 * Emission[n, Observation[t]])

        P = np.sum(F[:, T - 1])

        return P, F
    except Exception:
        return None, None
