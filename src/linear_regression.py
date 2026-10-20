import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.theta_0 = 0.0
        self.theta_1 = 0.0
        self.cost_history = []

        self.x_mean = 0.0
        self.x_std = 1.0

    def fit(self, X, y):
        # Ensure X and y are NumPy arrays
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        m = len(X)

        if m == 0 or len(y) != m:
            raise ValueError("X and y must have the same non-zero length.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")

        # Scale X for stable gradient descent
        self.x_mean = np.mean(X)
        self.x_std = np.std(X)

        if self.x_std == 0:
            self.x_std = 1.0

        X_scaled = (X - self.x_mean) / self.x_std

        # Reset parameters when training
        self.theta_0 = 0.0
        self.theta_1 = 0.0
        self.cost_history = []

        for _ in range(self.epochs):
            y_pred = self.theta_0 + self.theta_1 * X_scaled
            error = y_pred - y

            # Calculate gradients
            grad_0 = np.mean(error)
            grad_1 = np.mean(error * X_scaled)

            # Update parameters
            self.theta_0 -= self.lr * grad_0
            self.theta_1 -= self.lr * grad_1

            # Record cost
            cost = np.mean(error ** 2) / 2
            self.cost_history.append(cost)

        # Convert parameters back to the original X scale
        self.theta_1 = self.theta_1 / self.x_std
        self.theta_0 = self.theta_0 - (
            self.theta_1 * self.x_mean
        )

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.theta_0 + self.theta_1 * X

    def get_params(self):
        return self.theta_0, self.theta_1