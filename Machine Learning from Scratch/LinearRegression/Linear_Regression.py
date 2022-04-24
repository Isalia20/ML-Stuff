import numpy as np


class LinearRegression:

    def __init__(self,
                 fit_intercept=True,
                 learning_rate=0.1,
                 steps=100
                 ):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.steps = steps
        self.weight_matrix = None

    def _initialize_weight_matrices(self, x):
        self.weight_matrix = np.random.normal(0, 0.1, (x.shape[1], 1))

    def _make_predictions(self, x):
        prediction = x @ self.weight_matrix

    def fit(self, x, y):
        self._initialize_weight_matrices(x)
        m = x.shape[0]
        y = y.reshape(-1, 1)
        for _ in range(self.steps):
            prediction = x @ self.weight_matrix
            derivative = (1/m) * ((prediction - y).T @ x).T  # shape of (1,m) by shape of (m,5).
            self.weight_matrix = self.weight_matrix - self.learning_rate * derivative
            if np.sum((1/m) * ((prediction - y)**2)) < 0.01:  # This means regression has converged.
                break

    def predict(self, x):
        y_pred = x @ self.weight_matrix
        return y_pred
