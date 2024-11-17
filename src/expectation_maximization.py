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
        """Computes group responsibilities for each observation.

        Args:
            data: Array of shape (n_samples, 3) with [user_idx, item_idx, rating]
            theta: User group memberships of shape (n_users, n_user_groups)
            eta: Item group memberships of shape (n_items, n_item_groups)
            pr: Rating probabilities of shape (n_user_groups, n_item_groups, n_ratings)

        Returns:
            Array of shape (n_samples, n_user_groups, n_item_groups) with ω values
        """

        user_indices = data[:, 0]
        item_indices = data[:, 1]
        rating_indices = data[:, 2]

        self._omegas[:] = (theta[user_indices][:, :, np.newaxis] *
                           eta[item_indices][:, np.newaxis, :] *
                           np.moveaxis(pr[:, :, rating_indices], -1, 0))

        return self._omegas

    def update_coefficients(self, data, theta, eta, pr):
        """Updates model parameters using fully vectorized operations.

        Implements equations 3-5 from the paper:
        θ_uk = Σ_i ω_ui(k,ℓ) / d_u
        η_iℓ = Σ_u ω_ui(k,ℓ) / d_i
        p_kℓ(r) = Σ_{ui|r_ui=r} ω_ui(k,ℓ) / Σ_ui ω_ui(k,ℓ)

        Args:
            data: Array of shape (n_samples, 3) with [user_idx, item_idx, rating]
            theta: Current user memberships
            eta: Current item memberships
            pr: Ratings probabilities

        Returns:
            Tuple (theta, eta, prs) with updated parameters
        """
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
        # Normalize using pre-computed factors
        return df / self._normalization_factors[type_]

    @staticmethod
    def normalize_with_self(df):
        """Normalizes probabilities in a 3D array along the last axis.

        Used to normalize rating probabilities p_kℓ(r) such that they sum to 1
        for each user group k and item group ℓ pair.

        Process:
        1. Reshapes 3D array (k,ℓ,r) to 2D array ((k*ℓ),r)
        2. Normalizes each row to sum to 1
        3. Reshapes back to original 3D shape

        Args:
            df: 3D numpy array of shape (n_user_groups, n_item_groups, n_ratings)
                Typically contains unnormalized rating probabilities.

        Returns:
            3D numpy array of same shape with normalized probabilities.
            For each k,ℓ pair: Σ_r p_kℓ(r) = 1

        Example:
            >>> pr = np.random.random((2, 3, 5))  # 2 user groups, 3 item groups, 5 ratings
            >>> pr_norm = normalize_with_self(pr)
            >>> assert np.allclose(pr_norm.sum(axis=2), 1)  # Sums to 1 for each k,ℓ

        Notes:
            - Handles zero-sum rows by replacing denominator with 1
            - Preserves original array shape
            - Used in EM algorithm for normalizing rating probabilities
        """
        temp = df.reshape((df.shape[0] * df.shape[1], df.shape[2]))
        return (
                temp / (np.where(temp.sum(axis=1) == 0, 1, temp.sum(axis=1)))[:, np.newaxis]
        ).reshape(df.shape)

    def compute_likelihood(self, data, theta, eta, pr):
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
