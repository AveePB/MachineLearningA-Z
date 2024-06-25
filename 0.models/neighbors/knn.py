from collections import Counter
import numpy as np

class KNNClassifier:
    
    def __init__(self, k= 5) -> None:
        self._k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._X_train = X
        self._y_train = y

    def __euclidean_distance(self, x1: np.ndarray, x2: np.ndarray):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X: np.ndarray):
        return np.array([self.__predict(x) for x in X])

    def __predict(self, x: np.ndarray):
        # compute the distances
        dist = np.array([self.__euclidean_distance(x, x_train) for x_train in self._X_train])

        # get the closest k
        k_indices = np.argsort(dist)[:self._k]
        neighbor_labels = self._y_train[k_indices]
        
        # majority vote
        return self.__most_common_label(neighbor_labels)

    def __euclidean_distance(self, x1: np.ndarray, x2: np.ndarray):
        return np.sqrt(np.sum((x1 - x2)**2))

    def __most_common_label(self, y: np.ndarray):
        return Counter(y).most_common(1)[0][0]