import numpy as np


class LinearRegression:

    def __init__(self,
                 fit_intercept=True,
                 learning_rate=0.1,  # learning rate for gradient descent
                 max_iter=100,  # Maximum iterations for gradient descent to take
                 tol=0.001,  # Loss after which fitting should stop
                 random_state=42  # Random state for reproducing results
                 ):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weight_matrix = None
        self.bias = 0
        self.random_state = random_state

    def _initialize_weight_matrices(self, x):
        np.random.seed(self.random_state)
        self.weight_matrix = np.random.normal(0, 0.1, (x.shape[1], 1))
        if self.fit_intercept:
            self.bias = np.random.normal(0, 0.1)

    def fit(self, x, y):
        self._initialize_weight_matrices(x)
        m = x.shape[0]
        y = y.reshape(-1, 1)
        for _ in range(self.max_iter):
            prediction = x @ self.weight_matrix + self.bias
            derivative = (1/m) * ((prediction - y).T @ x).T  # shape of (1,m) by shape of (m,5).
            self.weight_matrix -= self.learning_rate * derivative
            # Updating bias term if fit intercept is True
            if self.fit_intercept:
                derivative_b = (1 / m) * np.sum(prediction - y)
                self.bias -= self.learning_rate * derivative_b
            # Stopping fitting if loss is too low
            if np.sum((1/m) * ((prediction - y)**2)) < self.tol:
                break

    def predict(self, x):
        y_pred = x @ self.weight_matrix + self.bias
        return y_pred
