import numpy as np

class LogisticRegressor:

    def __init__(self, lr = 0.001, n_iters = 1000) -> None:
        self.__lr = lr 
        self.__n_iters = n_iters

        self._coeff = None
        self._inter = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self._coeff = np.zeros(n_features)
        self._inter = 0

        f = lambda x: np.dot(x, self._coeff) + self._inter

        for _ in range(self.__n_iters):
            y_pred = self.__sigmoid(f(X))

            dc = (1/n_samples) * np.dot(X.T, (y_pred - y)) 
            di = (1/n_samples) * np.sum(y_pred - y)

            self._coeff = self._coeff - self.__lr * dc
            self._inter = self._inter - self.__lr * di

    def __sigmoid(self, z: np.ndarray):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X: np.ndarray):
        f = lambda x: np.dot(x, self._coeff) + self._inter
        y_pred = self.__sigmoid(f(X))

        return np.array([(0 if y < 0.5 else 1) for y in y_pred])