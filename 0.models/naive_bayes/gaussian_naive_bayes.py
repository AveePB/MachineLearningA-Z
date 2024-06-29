import numpy as np

class GaussianNB:
    
    def __init__(self) -> None:
        self._labels: np.ndarray = None
        self._mean = {}
        self._var = {}
        self._priors = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._labels = np.unique(y)
        self._mean = {}
        self._var = {}
        self._priors = {}
        
        for l in self._labels:
            X_l = X[y == l]
            
            # Do computations for each label
            self._mean[l] = np.mean(X_l, axis=0)
            self._var[l] = np.var(X_l, axis=0)
            self._priors[l] = X_l.size / X.size

    def predict(self, X: np.ndarray):
        return np.array([self.__predict(x) for x in X])
    
    def __predict(self, x: np.ndarray):
        posteriors = []

        for l in self._labels:
            # Class conditional probability 
            ccp = self.__gaussian_pdf(x, l)
            prior = self._priors[l]

            posterior = np.sum(np.log(ccp) + np.log(prior))
            posteriors.append(posterior)

        return self._labels[np.argmax(posteriors)]

    # Gaussian probability density function
    def __gaussian_pdf(self, x: np.ndarray, label):
        mean = self._mean[label]
        var = self._var[label]

        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator