Expectation-Maximization Algorithm
==================================

The Problem
-----------

We want to find the optimal parameters for our MMSBM model that best explain the observed ratings. These parameters are:

* :math:`\theta_{uk}`: Probability of user u belonging to group k
* :math:`\eta_{i\ell}`: Probability of item i belonging to group ℓ
* :math:`p_{k\ell}(r)`: Probability of rating r for user group k and item group ℓ

The Challenge
-------------

We can't directly optimize these parameters because we don't know which groups were "responsible" for each rating. For example, when user u rates item i, we don't know which of u's group memberships interacted with which of i's group memberships to produce that rating.

This is where EM comes in. It's an iterative algorithm that:

1. Guesses which groups were responsible for each rating (E-step)
2. Updates the parameters based on these guesses (M-step)

Key Variables and Notation
--------------------------

Model Parameters
^^^^^^^^^^^^^^^^

* :math:`\theta_{uk}`: User group memberships
    - Probability that user u belongs to group k
    - Must sum to 1 for each user: :math:`\sum_k \theta_{uk} = 1`
    - Implemented as ``theta`` in the code

* :math:`\eta_{i\ell}`: Item group memberships
    - Probability that item i belongs to group ℓ
    - Must sum to 1 for each item: :math:`\sum_\ell \eta_{i\ell} = 1`
    - Implemented as ``eta`` in the code

* :math:`p_{k\ell}(r)`: Rating probabilities
    - For each user group k and item group ℓ, probability of rating r
    - Must sum to 1 for each (k,ℓ): :math:`\sum_r p_{k\ell}(r) = 1`
    - Implemented as ``pr`` in the code

Helper Variables
^^^^^^^^^^^^^^^^

* :math:`R^O`: Set of observed ratings
    - All (user, item, rating) triplets in our training data
    - Implemented as ``train`` in the code

* :math:`d_u`: User degree
    - Number of items rated by user u
    - Used for normalization
    - Stored in ``self._normalization_factors['user']``

* :math:`d_i`: Item degree
    - Number of users who rated item i
    - Used for normalization
    - Stored in ``self._normalization_factors['item']``

* :math:`\omega_{ui}(k,\ell)`: Group responsibilities
    - Probability that rating :math:`r_ui` was due to user group :math:`k` and item group :math:`l`
    - Computed in E-step
    - Implemented in ``compute_omegas``

The Algorithm
-------------

Starting Point
^^^^^^^^^^^^^^

We begin with random initial values for :math:`\theta`, :math:`\eta`, and :math:`p`, properly normalized::

    # Initialize random parameters
    theta = rng.random((n_users, n_user_groups))
    eta = rng.random((n_items, n_item_groups))
    pr = rng.random((n_user_groups, n_item_groups, n_ratings))

    # Normalize them
    theta = normalize_with_d(theta, d0)  # Sum to 1/du for each user
    eta = normalize_with_d(eta, d1)      # Sum to 1/di for each item
    pr = normalize_with_self(pr)         # Sum to 1 for each (k,ℓ)

Expectation Step (E-step)
^^^^^^^^^^^^^^^^^^^^^^^^^

In this step, we compute :math:`\omega_{ui}(k,\ell)`: our best guess for which groups were responsible for each rating.

.. math::
   \omega_{ui}(k,\ell) = \frac{\theta_{uk} \eta_{i\ell} p_{k\ell}(r_{ui})}{\sum_{k',\ell'} \theta_{uk'} \eta_{i\ell'} p_{k'\ell'}(r_{ui})}

This is like saying: "Given the current parameters, what's the probability that user u was acting as a member of group k and item i as a member of group ℓ when this rating was made?"

Implementation::

    def compute_omegas(self, data, theta, eta, pr):
        # Get indices for vectorization
        user_indices = data[:, 0]
        item_indices = data[:, 1]
        rating_indices = data[:, 2]

        # Compute numerator of omega
        self._omegas[:] = (theta[user_indices][:, :, np.newaxis] *
                          eta[item_indices][:, np.newaxis, :] *
                          np.moveaxis(pr[:, :, rating_indices], -1, 0))

        # Normalize to get probabilities
        sum_omega = np.sum(self._omegas, axis=(1, 2))
        return np.divide(self._omegas, sum_omega[:, np.newaxis, np.newaxis])

Maximization Step (M-step)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we use our :math:`\omega` values to update the parameters. We're maximizing the likelihood: the probability of observing our actual ratings given the model parameters.

1. Update user memberships::

    # For each user u and group k
    theta_uk = sum(omega_ui(k,ℓ) for i,ℓ in user_u_ratings) / d_u

This says: "If we believe omega_ui(k,ℓ) is the probability that rating rui came from groups (k,ℓ), then theta_uk should be proportional to how often user u used group k."

Implementation::

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


Parameter Normalization
-----------------------

All parameters must be normalized to represent valid probability distributions:

1. User normalization: :math:`\sum_k \theta_{uk} = 1`
   Implemented in ``normalize_with_d``::

    def normalize_with_d(self, df, type_):
        """Normalize using pre-computed factors"""
        return df / self._normalization_factors[type_]

2. Item normalization: :math:`\sum_\ell \eta_{i\ell} = 1`
   Uses the same function.

3. Rating normalization: :math:`\sum_r p_{k\ell}(r) = 1`
   Implemented in ``normalize_with_self``::

    def normalize_with_self(df):
        temp = df.reshape((df.shape[0] * df.shape[1], df.shape[2]))
        return (
                temp / (np.where(temp.sum(axis=1) == 0, 1, temp.sum(axis=1)))[:, np.newaxis]
        ).reshape(df.shape)

Log-likelihood
--------------

The quantity reported as ``likelihood`` (for every run in ``results`` and in ``score()``) is the log-likelihood of the training ratings under the fitted parameters,

.. math::
   \log L = \sum_{(u,i)} \log p(r_{ui} \mid u, i) = \sum_{(u,i)} \log \sum_{k,\ell} \theta_{uk}\, \eta_{i\ell}\, p_{k\ell}(r_{ui}),

which the EM iterations never decrease. With ``sampling > 1`` it decides which run's parameters are exposed; the predictions average all runs.

.. versionchanged:: 1.2.0
   Before 1.2.0 the reported value was the negative weighted entropy of the responsibilities, :math:`\sum_{(u,i)} \sum_{k,\ell} \tilde\omega_{ui}(k,\ell) \log \omega_{ui}(k,\ell)` with :math:`\tilde\omega` the unnormalised products :math:`\theta_{uk}\, \eta_{i\ell}\, p_{k\ell}(r_{ui})`, which is not a log-likelihood.

The degree-corrected M-step
---------------------------

With ``degree_corrected=True`` the E-step is unchanged: the responsibilities :math:`\omega_{ui}(k,\ell)` are computed by the same kernels, because the Poisson model attaches latent groups only to the observed ratings. The expected complete-data log-likelihood becomes

.. math::
   Q = \sum_{(u,i)} \sum_{k,\ell} \omega_{ui}(k,\ell) \log\big(\theta_{uk}\eta_{i\ell}p_{k\ell}(r_{ui})\big) - \sum_{k,\ell,r} \Theta_k H_\ell\, p_{k\ell}(r),

and, in the degree gauge described in :doc:`mmsbm`, its maximisation has the closed form

.. math::
   \theta_{uk} = \sum_{i \in \partial u} \sum_\ell \omega_{ui}(k,\ell), \qquad
   \eta_{i\ell} = \sum_{u \in \partial i} \sum_k \omega_{ui}(k,\ell), \qquad
   p_{k\ell}(r) = \frac{\sum_{(u,i):\, r_{ui}=r} \omega_{ui}(k,\ell)}{\Theta_k H_\ell}.

Compared with the standard M-step, the membership numerators are simply *not* divided by the degrees (so the rows of θ and η sum to the degrees), and the rating probabilities are divided by the half-edge counts of the two groups instead of being normalised over ratings. The three updates are jointly stationary, so the Poisson log-likelihood never decreases from one iteration to the next. The extra cost is a handful of column sums.

Implementation::

    def m_step_degree_corrected(n_theta, n_eta, n_pr):
        half_edges_k = n_theta.sum(axis=0)       # Θ_k
        half_edges_l = n_eta.sum(axis=0)         # H_ℓ
        denom = half_edges_k[:, None, None] * half_edges_l[None, :, None]
        pr = np.divide(n_pr, denom, out=np.zeros_like(n_pr), where=denom > 0)
        return n_theta, n_eta, pr

The log-likelihood reported for the degree-corrected model is

.. math::
   \log L = \sum_{(u,i)} \log \sum_{k,\ell} \omega_{ui}(k,\ell) - \sum_{k,\ell,r} \Theta_k H_\ell\, p_{k\ell}(r),

where the second term equals the number of observed ratings right after an M-step.

Initialisation
^^^^^^^^^^^^^^

A uniform affinity between user and item groups is a stationary point of the Poisson likelihood: around it the membership update is marginally stable and any structure in the links can only grow at second order, so runs started there stay at the symmetric solution in which every user has the same membership up to its degree. The degree-corrected model therefore multiplies the random initial rating probabilities by a random affinity pattern in which every user group prefers one item group ten to one (``ExpectationMaximization.initial_affinity``). The random draws for θ, η and p are the same as in the standard model; only this extra factor is added.

Performance Optimizations
-------------------------

Several optimizations make this implementation efficient:

1. **Vectorization**: Operations are done on arrays rather than loops::

    omegas = theta[user_indices][:, :, np.newaxis] * eta[item_indices][:, np.newaxis, :]

2. **Pre-computation**: The degrees that normalise θ and η are computed once, in linear time::

    user_degrees = np.bincount(train[:, 0], minlength=n_users)

3. **No process pool for a single run**: with ``sampling=1`` the EM runs in the calling process; worker processes are only started when several runs are requested.

Want to Learn More?
-------------------

* Check out the :doc:`../../api/modules` for detailed API documentation
* See the :doc:`mmsbm` for a description of the MMSBM algorithm
* Look at :doc:`../../guides/quickstart` for practical examples