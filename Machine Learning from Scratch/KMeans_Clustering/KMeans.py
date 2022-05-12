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
        self.centroids = None

    def initialize_clusters(self, x):
        np.random.seed(self.random_state)
        rng = default_rng()
        rand_ints_for_clusters = rng.choice(x.shape[0], size=self.n_clusters, replace=False)
        centroids = x[rand_ints_for_clusters, :]
        return centroids

    def _initialize_cluster_distance_matrix(self, x):
        cluster_distance_matrix = np.zeros(shape=(x.shape[0], self.n_clusters))
        return cluster_distance_matrix

    @staticmethod
    def _get_distance_vector_for_clusters(cluster_distance_matrix, centroids, centroid_index, x):
        centroid_coordinates = centroids[centroid_index]
        distance_matrix = x - centroid_coordinates
        distance_vector = np.sum(distance_matrix, axis=1) ** 2
        cluster_distance_matrix[:, centroid_index] = distance_vector
        return cluster_distance_matrix

    def fit(self, x):
        centroids = self.initialize_clusters(x)
        cluster_distance_matrix = self._initialize_cluster_distance_matrix(x)
        # Loop through max iterations to get the clusters
        for iteration in range(self.max_iter):
            # Get cluster distance matrix so we can then get the minimum distance which will be used
            # for assigning cluster for a sample.
            for centroid in range(self.n_clusters):
                cluster_distance_matrix = self._get_distance_vector_for_clusters(cluster_distance_matrix, centroids,
                                                                                 centroid, x)
            # Cluster index for the minimum distance per sample to the centroids
            cluster_indices = np.argmin(cluster_distance_matrix, axis=1)
            # Adjusting centroid coordinates.
            for centroid in range(self.n_clusters):
                cluster_index = np.where(cluster_indices == centroid)[0]
                x_cluster = x[cluster_index, :]
                centroids[centroid, :] = np.mean(x_cluster, axis=0, keepdims=True)
        # Saving centroids
        self.centroids = centroids

    def predict(self, x):
        # Initialize cluster distance matrix
        cluster_distance_matrix = self._initialize_cluster_distance_matrix(x)
        # Get distance vector for each cluster for each sample
        for centroid in range(self.n_clusters):
            cluster_distance_matrix = self._get_distance_vector_for_clusters(cluster_distance_matrix, self.centroids,
                                                                             centroid, x)
        # Get which clusters X belongs to.
        cluster_indices = np.argmin(cluster_distance_matrix, axis=1)
        return cluster_indices
