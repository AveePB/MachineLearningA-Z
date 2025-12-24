import numpy as np

class SupportVectorClassifier:
    
    def __init__(self, lr = 0.001, lambda_param = 0.01, n_iters = 1000) -> None:
        self._lr = lr
        self._lambda = lambda_param
        self._n_iters = n_iters

        self._coeff = None
        self._inter = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        f = lambda x: np.dot(x, self._coeff) - self._inter
        self._coeff = np.zeros(n_features)
        self._inter = 0

        # Gradient Descent
        for _ in range(self._n_iters):
            for i, x in enumerate(X):
                # Correct classification
                if (y_[i] * f(x) >= 1):
                    self._coeff -= self._lr * (2 * self._lambda * self._coeff)
                # Misclassification
                else:
                    self._coeff -= self._lr * (2 * self._lambda * self._coeff - np.dot(x, y_[i]))
                    self._inter -= self._lr * y[i]

    def predict(self, X: np.ndarray):
        f = lambda x: np.dot(x, self._coeff) - self._inter
        return np.sign(f(X))
