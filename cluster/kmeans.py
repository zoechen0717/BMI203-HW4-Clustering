import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # Check the inputs
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Number of clusters (k) is wrong")
        if tol <= 0:
            raise ValueError("Tolerance need to be positive")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("Maximum iterations is wrong")

        # Initialize parameters
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.error = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        #Check the input format
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input data is wrong.")
        if mat.shape[0] < self.k:
            raise ValueError("Number of observations < k")

        # Randomly initialize centroids
        # Initialize centroids are inspired by Isobel's GMM tutorial:https://github.com/IJbeasley/GMM-Tutorial/blob/main/GMM_tutorial.ipynb
        np.random.seed(123)
        self.centroids = mat[np.random.choice(mat.shape[0], self.k, replace=False)]

        # Lloyd’s Algorithm psedocode inspired by the class lecture and this princeton cs lecture notes:https://www.cs.princeton.edu/courses/archive/spring19/cos324/files/kmeans.pdf
        # For loop to find the optimimal clustering
        for _ in range(self.max_iter):
            # Compute distances
            dis = cdist(mat, self.centroids, metric='euclidean')
            # Assign point to the nearest centroids
            self.labels = np.argmin(dis, axis=1)
            # Compute new centroids
            new_centroids = np.array([mat[self.labels == i].mean(axis=0) for i in range(self.k)])
            # Check for covergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        # Compute final clustering errorß
        self.error = np.mean(np.min(cdist(mat, self.centroids, metric='euclidean'), axis=1) ** 2)

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Ensure model is trained
        if self.centroids is None:
            raise ValueError("The model hasn't been trained")

        # Check input matrix
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input data is wrong.")
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Feature mismatched")

        # The predict and fit parts also inspired by this pyton tutorial:https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/
        # Compute distances and assign points to nearest centroid
        distances = cdist(mat, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
                # Ensure model has been trained
        if self.error is None:
            raise ValueError("The model hasn't been fitted yet.")
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
                # Ensure model has been trained
        if self.centroids is None:
            raise ValueError("The model hasn't been fitted yet.")
        return self.centroids
