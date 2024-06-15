import numpy as np

class MultipleLinearRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.__lr = lr
        self.__n_iters = n_iters

        self._coeff = None
        self._inter = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._coeff = np.zeros(n_features)
        self._inter = 0

        for _ in range(self.__n_iters):
            y_pred = np.dot(X, self._coeff) + self._inter

            dc = (1/n_samples) * np.dot(X.T, (y_pred - y))
            di = (1/n_samples) * np.sum(y_pred - y)

            self._coeff = self._coeff - self.__lr * dc
            self._inter = self._inter - self.__lr * di

    def predict(self, X: np.ndarray):
        return np.dot(X, self._coeff) + self._inter
