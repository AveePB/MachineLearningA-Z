from collections import Counter
import numpy as np

class DecisionTreeClassification:

    class Node:

        def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
            self._feature = feature
            self._threshold = threshold
            self._value = value

            self._left = left
            self._right = right

        def is_leaf(self):
            return self._value is not None


    def __init__(self, min_samples = 2, max_depth = 100):
        self._min_samples = min_samples
        self._max_depth = max_depth

        self._n_features = None
        self._root = None

    def fit(self, X: np.ndarray , y: np.ndarray):
        self._n_features = X.shape[1]
        self._root = self.__grow_tree(X, y)

    def __grow_tree(self, X: np.ndarray, y: np.ndarray, *, depth = 0):
        n_labels = len(np.unique(y))
        n_samples, n_feats = X.shape

        # check stopping criteria
        if ((n_labels == 1) or (depth >= self._max_depth) or (n_samples < self._min_samples)):
            return DecisionTreeClassification.Node(value=self.__most_common_label(y))

        features = np.random.choice(n_feats, self._n_features, replace = False)

        # find best split
        best_feature, best_threshold = self.__best_split(X, y, features)

        # create children
        l_idxs, r_idxs = self.__split(X[:, best_feature], best_threshold)
        left = self.__grow_tree(X[l_idxs, :], y[l_idxs], depth=depth+1)
        right = self.__grow_tree(X[r_idxs, :], y[r_idxs], depth=depth+1)
        return DecisionTreeClassification.Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def __best_split(self, X: np.ndarray, y: np.ndarray, features: np.ndarray):
        best_variance, best_feature, best_threshold = -1, None, None
        parent_entropy = self.__entropy(y)

        for f in features:
            X_col = X[:, f]
            thresholds = np.unique(X_col)

            for t in thresholds:
                curr_variance = self.__compute_variance(y, X_col, t, parent_entropy)

                if (curr_variance > best_variance):
                    best_variance = curr_variance
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold

    def __compute_variance(self, y: np.ndarray, X_col: np.ndarray, treshold, parent_entropy):
        #create dummy children
        l_idxs, r_idxs = self.__split(X_col, treshold)

        if (len(l_idxs) == 0 or len(r_idxs) == 0): return 0

        #calculate weighted avg. entropy
        n, n_l, n_r = len(y), len(l_idxs), len(r_idxs)
        e_l, e_r = self.__entropy(y[l_idxs]), self.__entropy(y[r_idxs])

        children_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        return parent_entropy - children_entropy

    def __split(self, X_col: np.ndarray, treshold):
        left = np.argwhere(X_col <= treshold).flatten()
        right = np.argwhere(X_col > treshold).flatten()
        return left, right
    
    def __entropy(self, y: np.ndarray):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def __most_common_label(self, y: np.ndarray):
        return Counter(y).most_common(1)[0][0]        

    def predict(self, X: np.ndarray):
        return np.array([self.__traverse_tree(x, self._root) for x in X])

    def __traverse_tree(self, x: np.ndarray, root: Node):
        if (root.is_leaf()): return root._value

        if (x[root._feature] <= root._threshold):
            return self.__traverse_tree(x, root._left)

        return self.__traverse_tree(x, root._right)        
        