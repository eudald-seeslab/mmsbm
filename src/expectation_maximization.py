# expectation_maximization.py

from backend import load_backend
import numpy as np


class ExpectationMaximization:
    def __init__(self, dims, user_indices, item_indices, rating_indices,
                 norm_factors, backend: str = "auto", debug: bool = False):
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
        backend : str, optional
            Backend to use for computation ('numba' for Numba acceleration, 'numpy' for pure NumPy, or 'auto' to choose based on availability)
        """
        self._dims = dims
        self._user_indices = user_indices
        self._item_indices = item_indices
        self._rating_indices = rating_indices
        self._normalization_factors = norm_factors
        self._debug = debug

        # Load computational kernels dynamically
        (self._compute_omegas,
         self._update_coeffs,
         self._prod_dist,
         self._backend) = load_backend(backend)

        if self._debug:
            print(f"Using {self._backend} backend")

        # Pre-allocate arrays for results
        self._omegas = np.zeros((dims['n_samples'],
                                 dims['n_user_groups'],
                                 dims['n_item_groups']))

    def compute_omegas(self, data, theta, eta, pr):
        """Compute unnormalised responsibilities ω_{ui}(k,ℓ).

        For every observation (u, i, r) in *data* we compute the joint
        contribution of user-group *k* and item-group *ℓ*:

            ω_{ui}(k,ℓ) = θ_{uk} · η_{iℓ} · p_{kℓ}(r)

        The array returned therefore has shape ``(N, K, L)`` where *N* is the
        number of rows in *data*, *K* is the number of user groups and *L* the
        number of item groups.

        Parameters
        ----------
        data : ndarray of shape (N, 3)
            Each row contains encoded ``[user_idx, item_idx, rating_idx]``.
        theta : ndarray of shape (U, K)
            Membership distribution of each user over user groups.
        eta : ndarray of shape (I, L)
            Membership distribution of each item over item groups.
        pr : ndarray of shape (K, L, R)
            Probability of each rating given a pair of latent groups.

        Returns
        -------
        ndarray of shape (N, K, L)
            Unnormalised responsibilities ω.  They do *not* yet sum to one
            over all (k,ℓ) for a given (u,i) pair; that normalisation is
            performed later in the M-step.
        """

        # Delegate to selected backend kernel
        return self._compute_omegas(data, theta, eta, pr)

    def update_coefficients(self, data, theta, eta, pr):
        """M-step update of θ, η and p tensors.

        Implements the MMSBM update equations:

            θ_{uk}      = Σ_i Σ_ℓ ω_{ui}(k,ℓ) / d_u
            η_{iℓ}      = Σ_u Σ_k ω_{ui}(k,ℓ) / d_i
            p_{kℓ}(r)   = Σ_{ui | r_{ui}=r} ω_{ui}(k,ℓ)

        The accumulations are performed with scatter-add operations
        (``numpy.add.at``) to avoid building dense one-hot indicator
        matrices.

        Parameters
        ----------
        data : ndarray of shape (N, 3)
            Encoded observations ``[user_idx, item_idx, rating_idx]``.
        theta : ndarray of shape (U, K)
            Current estimate of user-group memberships.
        eta : ndarray of shape (I, L)
            Current estimate of item-group memberships.
        pr : ndarray of shape (K, L, R)
            Current estimate of rating probabilities.

        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            ``n_theta`` of shape (U, K), ``n_eta`` of shape (I, L) and
            ``n_pr`` of shape (K, L, R) containing the *unnormalised* updated
            numerators for each parameter.  They must be normalised by the
            caller before the next EM iteration.
        """

        return self._update_coeffs(data, theta, eta, pr)

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
        """Vectorised computation of product distributions for all data points.

        For each observation (u, i) we want:

            p(r|u,i) = Σ_k Σ_ℓ θ_{uk} η_{iℓ} p_{kℓ}(r)

        Using Einstein summation this becomes an efficient batched tensor
        contraction without explicit Python loops.
        """
        return self._prod_dist(data, theta, eta, pr)
