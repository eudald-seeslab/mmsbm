# expectation_maximization.py

import numpy as np

# -----------------------------------------------------------------------------
# Optional Numba acceleration
# -----------------------------------------------------------------------------

try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)
    def _compute_omegas_nb(data, theta, eta, pr):
        """Numba-accelerated version of compute_omegas (CPU)."""
        n = data.shape[0]
        K = theta.shape[1]
        L = eta.shape[1]

        omegas = np.empty((n, K, L))

        for idx in prange(n):
            u = data[idx, 0]
            i = data[idx, 1]
            r = data[idx, 2]
            for k in range(K):
                for l in range(L):
                    omegas[idx, k, l] = theta[u, k] * eta[i, l] * pr[k, l, r]

        return omegas

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _compute_omegas_nb = None
    NUMBA_AVAILABLE = False


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

        # Decide backend: 'numba' if requested/available, else 'numpy'
        if backend == "numba" and not NUMBA_AVAILABLE:
            raise ImportError("Numba backend requested but numba is not installed.")

        if backend == "numba" or (backend == "auto" and NUMBA_AVAILABLE):
            self._backend = "numba"
        else:
            self._backend = "numpy"

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

        user_idx = data[:, 0]
        item_idx = data[:, 1]
        rating_idx = data[:, 2]

        # Fast path: Numba kernel when available/selected
        if self._backend == "numba":
            return _compute_omegas_nb(data, theta, eta, pr)

        # ---------- NumPy fallback ----------

        pr_T = pr.transpose(2, 0, 1)  # (R, K, L)

        self._omegas[:] = (
            theta[user_idx][:, :, None] *
            eta[item_idx][:, None, :] *
            pr_T[rating_idx]
        )

        return self._omegas

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

        # --- E-step: compute responsibilities -----------------------------
        omegas = self.compute_omegas(data, theta, eta, pr)
        sum_omega = omegas.sum(axis=(1, 2))                     # (N,)

        eps = np.finfo(float).eps
        increments = omegas / np.maximum(sum_omega, eps)[:, None, None]

        user_idx = data[:, 0]
        item_idx = data[:, 1]
        rating_idx = data[:, 2]

        K = self._dims['n_user_groups']
        L = self._dims['n_item_groups']
        R = self._dims['n_ratings']

        # --- θ update -----------------------------------------------------
        inc_theta = increments.sum(axis=2)                      # (N, K)
        n_theta = np.zeros((theta.shape[0], K))
        np.add.at(n_theta, user_idx, inc_theta)

        # --- η update -----------------------------------------------------
        inc_eta = increments.sum(axis=1)                        # (N, L)
        n_eta = np.zeros((eta.shape[0], L))
        np.add.at(n_eta, item_idx, inc_eta)

        # --- p update -----------------------------------------------------
        n_pr = np.zeros((K, L, R))
        for r in range(R):
            mask = rating_idx == r
            if np.any(mask):
                n_pr[:, :, r] = increments[mask].sum(axis=0)

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
        """Vectorised computation of product distributions for all data points.

        For each observation (u, i) we want:

            p(r|u,i) = Σ_k Σ_ℓ θ_{uk} η_{iℓ} p_{kℓ}(r)

        Using Einstein summation this becomes an efficient batched tensor
        contraction without explicit Python loops.
        """
        user_idx = data[:, 0]
        item_idx = data[:, 1]

        theta_u = theta[user_idx]  # (N, K)
        eta_i = eta[item_idx]      # (N, L)

        # einsum: (N,K) , (N,L) , (K,L,R) -> (N,R)
        return np.einsum('nk,nl,klr->nr', theta_u, eta_i, pr)
