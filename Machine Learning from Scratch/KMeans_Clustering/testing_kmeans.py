import numpy as np
from KMeans_Clustering.KMeans import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns

# Generating data
X, y = make_blobs(n_samples = 1000,centers = 3, n_features=2,cluster_std= 0.1)
sns.scatterplot(X[:,0], X[:,1], hue = y)

# fitting kmeans
kmeans = KMeans(n_clusters=3, max_iter= 10000, random_state=42)
centroids = kmeans.fit(X)
# Plotting centroids to make sure it works
sns.scatterplot(kmeans.centroids[:,0], kmeans.centroids[:,1], hue = [0,1,2], palette = ["red","green","purple"])


# Checking the predict function
kmeans.predict(X)[0]

for i in range(3):
    print(np.sum((X[0] - kmeans.centroids[i])**2))
