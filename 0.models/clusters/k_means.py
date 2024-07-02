import numpy as np

def euclidean_distance(x1: np.ndarray, x2: np.ndarray):
    return np.sqrt(np.sum((x1 - x2)**2))

def assignToCluster(centroids: np.ndarray, x: np.array):
    dist = np.array([euclidean_distance(centroids[i], x) for i in range(len(centroids))])
    return np.argmin(dist)


class KMeans:

    def __init__(self, n_clusters = 3, n_iters = 1000) -> None:
        self.__n_clusters = n_clusters
        self.__n_iters = n_iters

        self.__centroids: np.ndarray = None
    
    def fit(self, X: np.ndarray):
        self.__centroids = X[np.random.choice(len(X), self.__n_clusters, replace=False)]
        
        for _ in range(self.__n_iters):
            clusters = [[] for _ in range(self.__n_clusters)]
            
            for x in X:
                y = assignToCluster(self.__centroids, x)
                clusters[y].append(x)
            

            for y in range(self.__n_clusters):
                if (len(clusters[y]) != 0):
                    self.__centroids[y] = np.mean(np.array(clusters[y]), axis=0)


    def predict(self, X: np.ndarray):
        return np.array([assignToCluster(self.__centroids, x) for x in X])