from PCA_Dimension_Reduction.PCA import PCA
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
# generate data
X = np.random.normal(0, 10, (100, 10))
X[:, 2] = 3 * X[:, 0] - 2 * X[:, 1] + np.random.normal(0, 0.3, 100)
X[:, 3] = 10 * X[:, 2] - 0.5 * X[:, 1] + np.random.normal(0, 0.3, 100)

pca = PCA(n_components=1)

pca.fit(X)
pca.transform(X).shape
