from tree.decision_tree_classification import DecisionTreeClassifier
from collections import Counter
import numpy as np

class RandomForestClassifier:

    def __init__(self, n_trees = 10, max_depth = 10, min_samples  = 2) -> None:
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._min_samples = min_samples

        self._n_features = None
        self._trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, self._n_features = X.shape
        self._trees = []

        for _ in range(self._n_trees):
            idxs = np.random.choice(n_samples, n_samples, replace = True)
        
            tree = DecisionTreeClassifier(self._min_samples, self._max_depth)
            tree.fit(X[idxs], y[idxs])
            self._trees.append(tree)        

    def predict(self, X: np.ndarray):
        predictions = np.array([tree.predict(X) for tree in self._trees])
        votes = np.swapaxes(predictions, 0, 1) # [[v1], [v2]], [[v1], [v2]] -> [[v1], [v1]], [[v2], [v2]]

        return np.array([self.__most_common_label(v) for v in votes])
    
    def __most_common_label(self, y: np.ndarray):
        return Counter(y).most_common(1)[0][0]
