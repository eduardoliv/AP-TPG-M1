#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logistic_regression.py
Baseline logistic regression with numpy only.
"""

import numpy as np

class LogisticRegressionModel:
    """
    Simple logistic regression using batch gradient descent.
    """

    def __init__(self, epochs=100, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            # Forward pass
            linear = X.dot(self.weights) + self.bias
            preds = 1.0 / (1.0 + np.exp(-linear))  # sigmoid

            # Gradient
            error = preds - y
            grad_w = (1.0 / n_samples) * X.T.dot(error)
            grad_b = (1.0 / n_samples) * np.sum(error)

            # Update
            self.weights -= self.lr * grad_w
            self.bias   -= self.lr * grad_b

    def predict_proba(self, X):
        linear = X.dot(self.weights) + self.bias
        return 1.0 / (1.0 + np.exp(-linear))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
