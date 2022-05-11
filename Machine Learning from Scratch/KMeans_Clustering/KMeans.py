# WIP
import numpy as np
from numpy.random import default_rng

class KMeans:
    def __init__(self,
                 n_clusters=3,
                 max_iter=100,
                 random_state=42
                 ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def _initialize_clusters(self, x):
        np.random.seed(self.random_state)
        rng = default_rng()
        rand_ints_for_clusters = rng.choice(x.shape[0], size=self.n_clusters, replace=False)
        centroids = x[rand_ints_for_clusters, :]
        return centroids

    def fit(self, x):
        centroids = self._initialize_clusters(x)
        cluster_distance_matrix = np.zeros(shape=(x.shape[0], self.n_clusters))
        for iteration in range(self.max_iter):
            print(iteration)
            for centroid in range(self.n_clusters):
                centroid_coordinates = centroids[centroid]
                distance_matrix = x - centroid_coordinates
                distance_vector = np.sum(distance_matrix, axis=1) ** 2
                cluster_distance_matrix[:, centroid] = distance_vector
            cluster_indices = np.argmin(cluster_distance_matrix, axis=1)

            for centroid in range(self.n_clusters):
                cluster_index = np.where(cluster_indices == centroid)[0]
                x_cluster = x[cluster_index, :]
                # return centroids
                # return centroids, x_cluster, cluster_indices
                centroids[:, centroid] = np.mean(x_cluster, axis=0, keepdims=True)
        return centroids






