# expectation_maximization.py

import numpy as np


class ExpectationMaximization:
    def __init__(self, dims, user_indices, item_indices, rating_indices, norm_factors):
        """
        Initialize EM algorithm with pre-computed values

        Parameters
        ----------
        dims : dict
            Dictionary with dimensions (n_samples, n_user_groups, n_item_groups, n_ratings)
        user_indices : list
            Pre-computed user indices for each user
        item_indices : list
            Pre-computed item indices for each item
        rating_indices : list
            Pre-computed rating indices for each rating
        norm_factors : dict
            Pre-computed normalization factors for users and items
        """
        self._dims = dims
        self._user_indices = user_indices
        self._item_indices = item_indices
        self._rating_indices = rating_indices
        self._normalization_factors = norm_factors

        # Pre-allocate arrays for results
        self._omegas = np.zeros((dims['n_samples'],
                                 dims['n_user_groups'],
                                 dims['n_item_groups']))

    def compute_omegas(self, data, theta, eta, pr):
        """Compute omegas using pre-allocated arrays"""
        user_indices = data[:, 0]
        item_indices = data[:, 1]
        rating_indices = data[:, 2]

        self._omegas[:] = (theta[user_indices][:, :, np.newaxis] *
                           eta[item_indices][:, np.newaxis, :] *
                           np.moveaxis(pr[:, :, rating_indices], -1, 0))

        return self._omegas

    def update_coefficients(self, data, theta, eta, pr):
        """Fully vectorized version"""
        omegas = self.compute_omegas(data, theta, eta, pr)
        sum_omega = np.zeros(self._dims['n_samples'])
        np.sum(omegas, axis=(1, 2), out=sum_omega)

        increments = np.divide(omegas, sum_omega[:, np.newaxis, np.newaxis])

        # Create sparse matrices for user, item, and rating memberships
        n_users = theta.shape[0]
        n_items = eta.shape[0]
        n_ratings = self._dims['n_ratings']

        user_matrix = np.zeros((data.shape[0], n_users))
        user_matrix[np.arange(data.shape[0]), data[:, 0]] = 1

        item_matrix = np.zeros((data.shape[0], n_items))
        item_matrix[np.arange(data.shape[0]), data[:, 1]] = 1

        rating_matrix = np.zeros((data.shape[0], n_ratings))
        rating_matrix[np.arange(data.shape[0]), data[:, 2]] = 1

        # Compute updates using matrix multiplication
        n_theta = user_matrix.T @ increments.sum(axis=-1)
        n_eta = item_matrix.T @ increments.sum(axis=1)
        n_pr = np.tensordot(increments, rating_matrix, axes=([0], [0]))

        return n_theta, n_eta, n_pr

    def normalize_with_d(self, df, type_):
        """Normalize using pre-computed factors"""
        return df / self._normalization_factors[type_]

    @staticmethod
    def normalize_with_self(df):
        """Normalize 3D arrays"""
        temp = df.reshape((df.shape[0] * df.shape[1], df.shape[2]))
        return (
                temp / (np.where(temp.sum(axis=1) == 0, 1, temp.sum(axis=1)))[:, np.newaxis]
        ).reshape(df.shape)

    def compute_likelihood(self, data, theta, eta, pr):
        """Compute likelihood with handling of zeros"""
        omegas = self.compute_omegas(data, theta, eta, pr)
        sum_omega = np.zeros(self._dims['n_samples'])
        np.sum(omegas, axis=(1, 2), out=sum_omega)

        epsilon = np.finfo(float).eps
        safe_omegas = np.maximum(omegas, epsilon)
        safe_sums = np.maximum(sum_omega, epsilon)

        return np.sum(safe_omegas * np.log(safe_omegas) -
                      safe_omegas * np.log(safe_sums[:, np.newaxis, np.newaxis]))

    @staticmethod
    def prod_dist(x, theta, eta, pr):
        """Compute product distribution for a single data point"""
        return (
            (theta[x[0]][:, np.newaxis, np.newaxis] *
             (eta[x[1], :][:, np.newaxis] * pr))
            .sum(axis=0)
            .sum(axis=0)
        )

    def compute_prod_dist(self, data, theta, eta, pr):
        """Compute product distribution for all data points"""
        return np.array([self.prod_dist(a, theta, eta, pr) for a in data])
