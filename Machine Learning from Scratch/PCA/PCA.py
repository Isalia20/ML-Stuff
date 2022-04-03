#WIP
import numpy as np


class PCA:
    def __init__(self,
                 n_components=None,
                 random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    @staticmethod
    def _normalize_features(self, x):
        x = x - np.mean(x, axis=0)
        return x

    def _generate_covariance_matrix(self, x):
        x = self._normalize_features(self, x)

        mat_x = np.matrix(x)
        cov_matrix = np.matmul(mat_x.T, mat_x)
        cov_matrix = cov_matrix * 1 / x.shape[0]

        return cov_matrix

    def _get_eigenvectors(self, x):
        cov_matrix = self._generate_covariance_matrix(x)
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        return eigen_values, eigen_vectors

    def _calculate_principal_components(self, x):
        eigen_values, eigen_vectors = self._get_eigenvectors(x)
        order = np.argsort(eigen_values)[::-1]
        principal_components = eigen_vectors[:, order]
        return principal_components

    def fit(self, x):
        self._calculate_principal_components(x)

    def transform(self,x):
