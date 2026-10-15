Mixed Membership Stochastic Block Model
=======================================

Introduction to MMSBM
---------------------

What is MMSBM?
^^^^^^^^^^^^^^

Mixed Membership Stochastic Block Model (MMSBM) is a probabilistic model for recommendation systems that:

* Discovers latent groups in both users and items
* Allows users and items to belong to multiple groups simultaneously
* Models rating patterns between groups
* Makes predictions based on group interactions

Why use MMSBM?
^^^^^^^^^^^^^^

Traditional recommendation systems often assume users and items have fixed characteristics. However, in reality:

* A user might watch action movies on weekends but documentaries during the week
* A mountain can be both a climbing spot and a skiing destination
* A book might appeal to both science fiction fans and mystery readers

MMSBM captures this flexibility by allowing "mixed memberships" - both users and items can belong partially to multiple groups.

Model Structure
---------------

Core Components
^^^^^^^^^^^^^^^

1. **User Groups** (K groups):

   * Each user u has a membership vector :math:`\theta_u`

   * :math:`\theta_{uk}` = probability of user u belonging to group k

   * Example: A user might be 70% "action fan" and 30% "climbing movies fan"::

       user_memberships = {
           'action_group': 0.7,
           'climbing_group': 0.3
       }

2. **Item Groups** (L groups):

   * Each item i has a membership vector :math:`\eta_i`

   * :math:`\eta_{i\ell}` = probability of item i belonging to group ℓ

   * Example: A movie might be 60% "action" and 40% "drama"::

       movie_memberships = {
           'action_group': 0.6,
           'drama_group': 0.4
       }

3. **Rating Patterns** (for each group pair):

   * :math:`p_{k\ell}(r)` = probability of rating r between user group k and item group ℓ

   * Captures how different user groups tend to rate different item groups

   * Example: "action fans" might tend to rate "action movies" highly::

       rating_patterns = {
           ('action_fan', 'action_movie'): {
               5: 0.6,  # 60% chance of 5-star rating
               4: 0.3,  # 30% chance of 4-star rating
               3: 0.1   # 10% chance of 3-star rating
           }
       }

Mathematical Formulation
------------------------

The Rating Process
^^^^^^^^^^^^^^^^^^

When a user rates an item, the model assumes:

1. The user acts as a member of some group k
2. The item acts as a member of some group ℓ
3. The rating follows the pattern for groups k and ℓ

This gives us the probability of a rating r:

.. math::
   Pr[r_{ui} = r] = \sum_{k,\ell} \theta_{uk} \eta_{i\ell} p_{k\ell}(r)

Implementation::

    def prod_dist(x, theta, eta, pr):
        return (theta[x[0]][:, np.newaxis, np.newaxis] *
                (eta[x[1], :][:, np.newaxis] * pr)
               ).sum(axis=0).sum(axis=0)

Constraints
^^^^^^^^^^^

All vectors must represent valid probability distributions:

1. User memberships sum to 1:

   .. math::
      \sum_k \theta_{uk} = 1 \quad \text{for all } u

2. Item memberships sum to 1:

   .. math::
      \sum_\ell \eta_{i\ell} = 1 \quad \text{for all } i

3. Rating probabilities sum to 1:

   .. math::
      \sum_r p_{k\ell}(r) = 1 \quad \text{for all } k,\ell

Implementation::

    def normalize_with_d(df, d):
        """Normalize user/item memberships"""
        return df / [np.repeat(max(len(a), 1), df.shape[1])
                    for a in d.values()]

    def normalize_with_self(df):
        """Normalize 3D arrays"""
        temp = df.reshape((df.shape[0] * df.shape[1], df.shape[2]))
        return (
                temp / (np.where(temp.sum(axis=1) == 0, 1, temp.sum(axis=1)))[:, np.newaxis]
        ).reshape(df.shape)

Model Training
--------------

The training process has two main components:

1. **Multiple Sampling Runs**:

   * Start with different random initializations

   * Run EM algorithm on each

   * Choose best result based on accuracy

2. **Cross-Validation Option**:

   * Split data into folds

   * Train on each subset

   * Test on held-out data

Implementation::

    def cv_fit(self, data, folds=5):
        """Cross-validated model fitting"""
        accuracies = []
        for f in range(folds):
            train, test = self._split_data(data, f)
            self.fit(train)
            acc = self.score(test)['accuracy']
            accuracies.append(acc)
        return accuracies


Making Predictions
------------------

To predict ratings for new user-item pairs:

1. Use learned group memberships (:math:`\theta`, :math:`\eta`)
2. Use learned rating patterns (:math:`p`)
3. Compute expected rating using the probability formula

Implementation::

    def predict(self, data):
        self._check_is_fitted()

        test = self.data_handler.format_test_data(data)
        self.test = test

        # Get the info for all the runs
        rats = [
            self.em.compute_prod_dist(test, a["theta"], a["eta"], a["pr"])
            for a in self.results
        ]
        prs = np.array([a["pr"] for a in self.results])
        likelihoods = np.array([a["likelihood"] for a in self.results])
        thetas = np.array([a["theta"] for a in self.results])
        etas = np.array([a["eta"] for a in self.results])

        best = self.choose_best_run(rats)

        # Store the cleaned best objects
        self.theta = self.data_handler.return_theta_indices(thetas[best])
        self.eta = self.data_handler.return_eta_indices(etas[best])
        self.pr = self.data_handler.return_pr_indices(prs[best])
        self.likelihood = likelihoods[best]

        # The prediction matrix is the average of all runs
        self.prediction_matrix = np.array(rats).mean(axis=0)

        return self.prediction_matrix


Interpreting Results
--------------------

The model provides several insights:

1. **User Groups**:

   * Which types of users exist?

   * How do different users combine these types?

   * Example: Discover "weekend watchers" vs "daily viewers"

2. **Item Groups**:

   * What are the natural categories?

   * How do items span multiple categories?

   * Example: Find movies that bridge genres

3. **Rating Patterns**:

   * How do different user types rate different item types?

   * Which combinations lead to high/low ratings?

   * Example: Understand what drives high ratings

Example Analysis::

    # Get group memberships for a user
    user_groups = model.theta.loc['user_123']
    print("User group memberships:", user_groups)

    # Get item categorization
    item_groups = model.eta.loc['item_456']
    print("Item group memberships:", item_groups)

    # Look at rating patterns
    rating_probs = model.pr
    print("Rating patterns:", rating_probs)


Degree-corrected variant
------------------------

Why a degree-corrected model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard MMSBM is *conditional on the observed pairs*: it models the probability of a rating given that user u has rated item i, and never asks which items a user rates or how many. This has two consequences:

* The number of ratings of a user or item (its *degree*) cannot influence the groups, because degrees never enter the likelihood.
* The structure of the bipartite graph of who-rates-what cannot influence the groups either. If two kinds of users rate different kinds of items but, once they rate, rate them in the same way, the standard model cannot tell them apart.

In the stochastic block model literature the second point is addressed by modelling the links themselves, and the first by *degree correction* (Karrer and Newman, 2011): every node receives a propensity parameter, so that the groups capture the pattern of connections beyond what the degrees already explain. The mixed-membership version of the degree-corrected model is due to Ball, Karrer and Newman (2011), and it is the overlapping model used in Peixoto's (2019) Bayesian treatment of block models. ``degree_corrected=True`` fits its bipartite, rating-valued form.

The model
^^^^^^^^^

Every cell of the user x item x rating tensor is a Poisson count with rate

.. math::
   \lambda_{uir} = \sum_{k,\ell} \theta_{uk}\, \eta_{i\ell}\, p_{k\ell}(r),

where θ and η are *unnormalised*: their row sums are free, so they absorb the degrees, and the groups are left to explain who connects to whom. The data are the cells with a rating (count one) and all the other cells (count zero), so the log-likelihood is, up to a constant,

.. math::
   \log L = \sum_{(u,i) \in R^O} \log \lambda_{u i r_{ui}} - \sum_{k,\ell,r} \Theta_k H_\ell\, p_{k\ell}(r),
   \qquad \Theta_k = \sum_u \theta_{uk},\; H_\ell = \sum_i \eta_{i\ell}.

The second term is the expected total number of ratings under the model and is what makes the unobserved pairs count. It only involves the column sums of θ and η, so the cost of an iteration is the same as for the standard model.

Interpretation of the parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rescaling :math:`\theta_{\cdot k}` by a constant and :math:`p_{k\ell}(r)` by its inverse leaves every λ unchanged, and likewise for η. The implementation fixes this freedom in the *degree gauge*, in which

* :math:`\sum_k \theta_{uk} = d_u`: the row of a user is its number of ratings split among the user groups. :math:`\theta_{uk}/d_u` is the membership distribution, comparable to the θ of the standard model, and :math:`\theta_{uk}` itself is the expected number of ratings that u makes as a member of group k (the labelled degree of Peixoto's overlapping model).
* :math:`\sum_\ell \eta_{i\ell} = d_i`, likewise for items.
* :math:`p_{k\ell}(r) = e_{k\ell}(r) / (e_k e_\ell)`, where :math:`e_{k\ell}(r)` is the expected number of ratings r between user group k and item group ℓ and :math:`e_k = \Theta_k`, :math:`e_\ell = H_\ell` are the numbers of ratings attached to each group. This is the usual form of the degree-corrected block matrix: :math:`\sum_r p_{k\ell}(r)` measures the affinity between the two groups relative to a random pairing of their ratings, and :math:`p_{k\ell}(r) / \sum_{r'} p_{k\ell}(r')` is the rating distribution of the pair, as in the standard model.

Predictions
^^^^^^^^^^^

Conditioning on the pair being observed recovers the quantity predicted by the standard model,

.. math::
   \Pr[r_{ui} = r \mid u \text{ rates } i] = \frac{\lambda_{uir}}{\sum_{r'} \lambda_{uir'}},

which is what ``predict`` returns, so accuracies and errors are comparable between the two models. In addition, the degree-corrected model predicts whether a pair is rated at all,

.. math::
   \Pr[u \text{ rates } i] = 1 - \exp\Big(-\sum_r \lambda_{uir}\Big),

available through ``predict_links``.

What to expect
^^^^^^^^^^^^^^

The degree-corrected model fits links and ratings jointly, so its groups reflect both who rates what (with popularity discounted) and how they rate. On data where the groups are only visible in the link pattern it recovers them and the standard model cannot; on data where the ratings carry all the signal both models find the same groups, but the degree-corrected one is not guaranteed to predict ratings better, because part of its parameters are spent on the links. Compare the two with ``held_out_log_likelihood`` or ``select_groups``.

A uniform affinity between groups is a stationary point of the Poisson likelihood from which the EM cannot escape, so the rating probabilities are initialised with a random 10:1 affinity pattern between user and item groups (see :doc:`em_algorithm`). As for the standard model, use ``sampling > 1`` to guard against local optima.

Choosing the number of groups
-----------------------------

``select_groups`` evaluates a grid of (user groups, item groups) by k-fold cross-validation and scores each combination by the mean held-out log-likelihood of the ratings,

.. math::
   \frac{1}{N_{\text{test}}} \sum_{(u,i) \in \text{test}} \log \Pr[r_{ui} \mid u, i],

which is defined identically for both models. Each user's ratings are dealt round-robin across the folds, so every user with at least two ratings stays in training, and test ratings whose user or item is absent from training are left out. A held-out criterion is used instead of a penalised training likelihood (AIC, BIC) because the number of membership parameters grows with the number of users and items, outside the regime in which those penalties are justified. Because the fits are maximum likelihood, the predicted distributions are smoothed towards the uniform distribution, :math:`p' = (1 - \alpha) p + \alpha / R` with :math:`\alpha = 0.05` by default, before taking logs: otherwise a handful of held-out ratings predicted with near-zero probability dominate the mean and favour too few groups. ``criterion="accuracy"`` selects by held-out accuracy instead; the grid reports both.

Want to Learn More?
-------------------

* See :doc:`em_algorithm` for optimization details
* Check :doc:`../../guides/quickstart` for practical examples
* Look at :doc:`../../api/modules` for API details