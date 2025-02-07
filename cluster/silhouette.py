import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # Check input
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input data must be 2D array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("Cluster labels must be 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of observations must match the number of labels.")

        # The silhousette calculation is based on the lecture slides and the detailed explaination from this modules documentation:https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/silhouette_score.html
        # Compute pairwise distances
        distances = cdist(X, X, metric="euclidean")
        silhouette_scores = np.zeros(X.shape[0])

        # For loop to calcuate the silhouette scores
        # 1. get mean distance to other points in the same cluster
        # 2. get the smallest mean distance to another cluster
        # 3. silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        # A good clustering with well separated and compact clusters will have a silhouette score close to 1. A low silhouette score (close to -1) indicates a poorly isolated cluster (both type I and type II error).
        for i in range(X.shape[0]):
            cluster_mask = (y == y[i])
            other_clusters_mask = ~cluster_mask

            # Compute a(i) - Mean distance to other points in the same cluster (excluding itself)
            a_i = np.mean(distances[i, cluster_mask & (np.arange(len(y)) != i)]) if np.sum(cluster_mask) > 1 else 0

            # Compute b(i) - Smallest mean distance to another cluster
            b_i = np.min([np.mean(distances[i, y == label]) for label in np.unique(y) if label != y[i]])

            # Compute silhouette score
            # Also handel edge cases of a_i and b_i
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return silhouette_scores
