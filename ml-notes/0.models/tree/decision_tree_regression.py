import numpy as np

class Node:
    
    def __init__(self, feature_idx = None, threshold = None, left = None, right = None, *, value = None):
        self._feature_idx = feature_idx
        self._threshold = threshold

        self._left: Node = left
        self._right: Node = right

        self._value = value
    
    def isLeaf(self):
        return self._value is not None


def calculate_mse(y_true: np.ndarray):
    # Current predicted value is mean of ys
    y_pred = np.mean(y_true)

    # Calculate metric
    return np.mean((y_true - y_pred)**2)

def best_split(X: np.ndarray, y: np.ndarray):
    best_mse, root, best_splits = float("inf"), None, None 
    n_samples, n_features = X.shape

    # Iterate through features
    for feat_idx in range(n_features):
        thresholds = np.unique(X[:, feat_idx])

        # Iterate through thresholds
        for thresh in thresholds:
            l_indices = X[:, feat_idx] <= thresh
            r_indices = X[:, feat_idx] > thresh

            # Skip, if only one group
            if (y[l_indices].size == 0 or y[r_indices].size == 0):
                continue
            
            # Compute weighted MSE
            mse_l = calculate_mse(y[l_indices])
            mse_r = calculate_mse(y[r_indices])
            mse = (y[l_indices].size * mse_l + y[r_indices].size * mse_r) / n_samples
            
            # Minimize MSE
            if (mse < best_mse):
                best_mse = mse
                best_splits = (l_indices, r_indices)
                root = Node(feat_idx, thresh)

    return root, best_splits


class DecisionTreeRegressor:
    
    def __init__(self, max_depth = 10, min_samples_split = 2) -> None:
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._root: Node = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._root = self.__grow_tree(X, y)

    def __grow_tree(self, X: np.ndarray, y: np.ndarray, depth = 0):
        # Out of bounds
        if ((y.size < self._min_samples_split) or (depth >= self._max_depth)): 
            return Node(value=np.mean(y))

        # Get best split
        root, best_splits = best_split(X, y)
        l_indices, r_indices = best_splits

        # Initialize node
        root._left = self.__grow_tree(X[l_indices], y[r_indices])
        root._right = self.__grow_tree(X[r_indices], y[r_indices])

        return root
    
    def predict(self, X: np.ndarray):
        return np.array([self.__traverse_tree(x) for x in X])

    def __traverse_tree(self, x: np.ndarray):
        curr_node = self._root

        while(not curr_node.isLeaf()):
            feat_idx, thresh = curr_node._feature_idx, curr_node._threshold
            # Go to left subtree
            if (x[feat_idx] <= thresh):
                curr_node = curr_node._left

            # Go to right subtree
            else:
                curr_node = curr_node._right
        
        return curr_node._value