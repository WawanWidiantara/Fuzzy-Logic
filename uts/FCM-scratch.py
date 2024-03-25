import numpy as np


class FuzzyCMeans:
    def __init__(self, n_clusters=5, m=3, max_iter=100, tolerance=0.00001):
        """
        Initializes the Fuzzy C-Means clustering algorithm.

        Args:
            n_clusters (int): The number of clusters to form.
            m (float): The fuzziness parameter (m > 1). Controls the degree of fuzziness in cluster assignments.
            max_iter (int): The maximum number of iterations.
            tolerance (float): Convergence threshold based on changes in the membership matrix.
        """

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centroids = None  # Cluster centroids
        self.U = None  # Membership matrix

    def fit(self, data):
        """
        Fits the Fuzzy C-Means model to the data.

        Args:
            data (numpy.ndarray): The input data.
        """

        # Initialize membership matrix with random values (normalized)
        self.U = np.random.rand(data.shape[0], self.n_clusters)
        self.U /= np.sum(self.U, axis=1)[:, np.newaxis]

        for _ in range(self.max_iter):
            # Update cluster centroids
            self.centroids = self._calculate_centroids(data)

            # Update membership degrees
            new_U = self._calculate_membership(data)

            # Check for convergence
            if np.linalg.norm(new_U - self.U) <= self.tolerance:
                break

            self.U = new_U

    def predict(self, data):
        """
        Predicts cluster assignments for new data points.

        Args:
            data (numpy.ndarray): The new data.

        Returns:
            numpy.ndarray: Cluster labels for each data point.
        """

        if self.centroids is None:
            raise ValueError("FCM model not fitted. Call fit() first.")
        return np.argmax(self._calculate_membership(data), axis=1)

    def _calculate_centroids(self, data):
        """
        Calculates cluster centroids based on current membership degrees.

        Args:
            data (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: Updated cluster centroids.
        """

        weighted_sum = np.dot((self.U**self.m).T, data)
        membership_sums = np.sum(self.U**self.m, axis=0)
        return weighted_sum / membership_sums[:, np.newaxis]

    def _calculate_membership(self, data):
        """
        Calculates membership degrees of data points to each cluster.

        Args:
            data (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The membership matrix.
        """

        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        distances[distances == 0] = np.finfo(float).eps  # Avoid division by zero

        inv_distances = 1 / (distances ** (2 / (self.m - 1)))
        return inv_distances / inv_distances.sum(axis=1)[:, np.newaxis]
