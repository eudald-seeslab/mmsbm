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

    # ------------------------------------------------------------------
    # Degree-corrected (Poisson) variant
    # ------------------------------------------------------------------

    @staticmethod
    def m_step_degree_corrected(n_theta, n_eta, n_pr):
        """M-step of the degree-corrected model, written in the *degree gauge*.

        The degree-corrected model (Ball, Karrer & Newman, 2011; Peixoto, 2019)
        treats every (user, item, rating) cell as a Poisson count with rate

            λ_{uir} = Σ_{kℓ} θ_{uk} η_{iℓ} p_{kℓ}(r),

        with θ and η *unnormalised*: their row sums absorb the node degrees.
        Rescaling θ_{·k} by c_k and p_{kℓ·} by 1/c_k (and likewise η_{·ℓ},
        d_ℓ) leaves every λ unchanged, so we may fix that freedom.  Choosing
        the gauge in which the expected degree of every node equals its
        observed degree turns the exact conditional maximisers of the EM
        objective into

            θ_{uk}    = Σ_{i∈∂u} Σ_ℓ ω_{ui}(k,ℓ)               (so Σ_k θ_{uk} = d_u)
            η_{iℓ}    = Σ_{u∈∂i} Σ_k ω_{ui}(k,ℓ)               (so Σ_ℓ η_{iℓ} = d_i)
            p_{kℓ}(r) = Σ_{ui: r_{ui}=r} ω_{ui}(k,ℓ) / (Θ_k H_ℓ),

        where Θ_k = Σ_u θ_{uk} and H_ℓ = Σ_i η_{iℓ} are the numbers of
        half-edges attached to each group.  In words: the membership
        numerators are *not* divided by the degrees, and the group-pair rates
        are normalised by the half-edge counts instead of over ratings.  This
        is the exact closed-form M-step (each block is the conditional
        maximiser, and the three blocks are jointly stationary), so the
        Poisson log-likelihood never decreases.

        Parameters
        ----------
        n_theta, n_eta, n_pr : ndarray
            Unnormalised numerators returned by :meth:`update_coefficients`.

        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            Updated ``theta`` (U, K), ``eta`` (I, L) and ``pr`` (K, L, R).
        """
        half_edges_k = n_theta.sum(axis=0)                      # Θ_k, shape (K,)
        half_edges_l = n_eta.sum(axis=0)                        # H_ℓ, shape (L,)
        denom = half_edges_k[:, None, None] * half_edges_l[None, :, None]
        pr = np.divide(n_pr, denom, out=np.zeros_like(n_pr), where=denom > 0)
        return n_theta, n_eta, pr

    @staticmethod
    def initial_affinity(rng, n_user_groups, n_item_groups, contrast=9.0):
        """Random affinity pattern used to initialise the degree-corrected model.

        A uniform affinity between user and item groups is a stationary point
        of the Poisson likelihood: the EM update of the memberships is then
        marginally stable and any link structure can only grow at second
        order, so runs started there stay at the symmetric solution (every
        user with the same mixed membership, up to its degree).  Multiplying
        the initial rating probabilities by a strong random pattern removes
        that degeneracy.  Each user group is assigned a preferred item group
        (a balanced random assignment, so that the pattern has full contrast
        for any K and L) and the corresponding rates are scaled by
        ``1 + contrast``.

        Returns
        -------
        ndarray of shape (K, L)
        """
        K, L = n_user_groups, n_item_groups
        preferred = np.concatenate([rng.permutation(L) for _ in range(K // L + 1)])[:K]
        pattern = np.ones((K, L))
        pattern[np.arange(K), preferred] += contrast
        return pattern

    def poisson_log_likelihood(self, data, theta, eta, pr):
        """Log-likelihood of the degree-corrected (Poisson) model.

        Up to an additive constant,

            log L = Σ_{(u,i,r) observed} log λ_{uir} − Σ_{kℓr} Θ_k H_ℓ p_{kℓ}(r),

        where the second term is the expected total number of ratings under
        the model, Σ_{u,i,r} λ_{uir}, computed in closed form from the column
        sums of θ and η.  Right after an M-step it equals the number of
        observations, so at convergence log L = Σ_obs log λ − N.

        Parameters
        ----------
        data : ndarray of shape (N, 3)
            Encoded observations ``[user_idx, item_idx, rating_idx]``.
        theta, eta, pr : ndarray
            Current parameters in the degree-corrected parametrisation.

        Returns
        -------
        float
        """
        omegas = self.compute_omegas(data, theta, eta, pr)
        rates = omegas.sum(axis=(1, 2))
        eps = np.finfo(float).eps
        expected_edges = np.einsum('k,l,klr->', theta.sum(axis=0), eta.sum(axis=0), pr)
        return np.sum(np.log(np.maximum(rates, eps))) - expected_edges

    @staticmethod
    def normalize_rows(x):
        """Normalise each row of a 2-D array to sum to one (all-zero rows stay zero).

        Used by the degree-corrected model to turn Poisson rates λ_{uir} into
        conditional rating probabilities p(r | u, i) = λ_{uir} / Σ_{r'} λ_{uir'}.
        """
        sums = x.sum(axis=1, keepdims=True)
        return np.divide(x, sums, out=np.zeros_like(x), where=sums > 0)

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
