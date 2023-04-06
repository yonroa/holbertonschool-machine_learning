#!/usr/bin/env python3
"""This module contains the class 'BayesianOptimization'"""

from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the class"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is False:
            mu_sample_opt = np.amax(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        else:
            mu_sample_opt = np.amin(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        with np.errstate(divide='warn'):
            if self.minimize is True:
                imp = mu_sample_opt - mu - self.xsi
            else:
                imp = mu - mu_sample_opt - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """optimizes the black-box function"""
        positions = []
        for i in range(iterations):
            next_x, ei = self.acquisition()
            next_y = self.f(next_x)
            pos = np.argmax(ei)
            if pos in positions:
                positions.append(np.argmax(ei))
                break
            self.gp.update(next_x, next_y)
            positions.append(np.argmax(ei))
        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        X_opt = self.gp.X[index]
        Y_opt = self.gp.Y[index]
        return X_opt, Y_opt
