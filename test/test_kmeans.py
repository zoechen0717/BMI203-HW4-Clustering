# Write your k-means unit tests here
import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from cluster import KMeans

def test_kmeans():
    # Generate test data
    np.random.seed(123)
    X = np.random.rand(100, 2) * 10  # 100 points in 2D space
    k = 3

    # Fit sklearn KMeans
    sklearn_kmeans = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
    sklearn_kmeans.fit(X)
    sklearn_labels = sklearn_kmeans.labels_
    sklearn_centroids = sklearn_kmeans.cluster_centers_

    # Fit custom KMeans
    custom_kmeans = KMeans(k=k)
    custom_kmeans.fit(X)
    custom_labels = custom_kmeans.predict(X)
    custom_centroids = custom_kmeans.get_centroids()

    # Check cluster assignments match up to permutation
    assert set(np.unique(custom_labels)) == set(np.unique(sklearn_labels))

    # Check centroids are close
    assert np.allclose(np.sort(custom_centroids, axis=0), np.sort(sklearn_centroids, axis=0), atol=1e-1)

    # Edge Cases
    with pytest.raises(ValueError):
        KMeans(k=-1)  # k must be > 0
    with pytest.raises(ValueError):
        KMeans(k=111).fit(X)  # k cannot exceed number of points
    with pytest.raises(ValueError):
        KMeans(k=3).fit(np.array([1, 2, 3]))  # Input must be 2D
