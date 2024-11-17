Mixed Membership Stochastic Block Model
====================================

Introduction to MMSBM
------------------

What is MMSBM?
^^^^^^^^^^^^

Mixed Membership Stochastic Block Model (MMSBM) is a probabilistic model for recommendation systems that:

* Discovers latent groups in both users and items
* Allows users and items to belong to multiple groups simultaneously
* Models rating patterns between groups
* Makes predictions based on group interactions

Why use MMSBM?
^^^^^^^^^^^^

Traditional recommendation systems often assume users and items have fixed characteristics. However, in reality:

* A user might watch action movies on weekends but documentaries during the week
* A mountain can be both a climbing spot and a skiing destination
* A book might appeal to both science fiction fans and mystery readers

MMSBM captures this flexibility by allowing "mixed memberships" - both users and items can belong partially to multiple groups.

Model Structure
-------------

Core Components
^^^^^^^^^^^^

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

   * Example: A movie might be 60% "action" and 40% "drama":

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
---------------------

The Rating Process
^^^^^^^^^^^^^^^

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
^^^^^^^^^

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
------------

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
---------------

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
-----------------

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


Want to Learn More?
----------------

* See :doc:`em_algorithm` for optimization details
* Check :doc:`../../guides/quickstart` for practical examples
* Look at :doc:`../../api/modules` for API details