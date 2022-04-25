import numpy as np


class LogisticRegression:

    def __init__(self,
                 fit_intercept=True,
                 learning_rate=0.1,  # learning rate for gradient descent
                 max_iter=100,  # Steps for gradient descent to take
                 tol=0.001,  # Loss after which fitting should stop
                 class_imbalance=2,  # Amount of times class 1 is lower than class 0 for gradient descent
                 threshold=0.5,  # Threshold for determining binary classification
                 random_state=42  # Random state for reproducing results
                 ):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.class_imbalance = class_imbalance
        self.threshold = threshold
        self.weight_matrix = None
        self.bias = 0
        self.random_state = random_state

    def _initialize_weight_matrices(self, x):
        np.random.seed(self.random_state)
        self.weight_matrix = np.random.normal(0, 0.1, (x.shape[1], 1))
        if self.fit_intercept:
            self.bias = np.random.normal(0, 0.1)

    @staticmethod
    def _sigmoid_function(z):
        a = 1/(1 + np.exp(-z))
        return a

    def fit(self, x, y):
        self._initialize_weight_matrices(x)
        m = x.shape[0]
        y = y.reshape(-1, 1)
        for i in range(self.max_iter):
            a = self._sigmoid_function(x @ self.weight_matrix + self.bias)
            derivative = (1/m) * (x.T @ (a - y))  # shape of (5,m) by shape of (m,1). Result is (5,1) shape of W
            self.weight_matrix -= self.learning_rate * derivative
            # Updating bias term if fit intercept is True
            if self.fit_intercept:
                derivative_b = (1/m) * np.sum(a - y)
                self.bias -= self.learning_rate * derivative_b
            # Stopping fitting if loss is too low, minus sign is not needed here as we compare it to pos value
            if abs(np.sum(y * np.log(a) - (1-y) * np.log(1-a))) < self.tol:
                print("minimum loss achieved on iteration " + str(i))
                break

    def predict_proba(self, x):
        y_pred = self._sigmoid_function(x @ self.weight_matrix + self.bias)
        return y_pred

    def predict(self, x):
        y_pred = self.predict_proba(x)
        return (y_pred >= self.threshold) * 1
