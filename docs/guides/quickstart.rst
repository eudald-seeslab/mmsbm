Quick Start Guide
=================

This guide will help you get started with MMSBM quickly.

Basic Usage
-----------

1. First, import the necessary modules::

    import pandas as pd
    from mmsbm import MMSBM

2. Prepare your data::

    # Example data format
    data = pd.DataFrame({
        'users': ['user1', 'user1', 'user2', 'user2'],
        'items': ['item1', 'item2', 'item1', 'item3'],
        'ratings': [5, 3, 4, 1]
    })

   The columns are found by these names, in any order, and other columns are
   ignored; other names are given with ``user_col``, ``item_col`` and
   ``rating_col``. A list of ``(user, item, rating)`` tuples or a 2-D array is
   read by position.

3. Initialize the model::

    model = MMSBM(
        user_groups=2,     # Number of user groups
        item_groups=4,     # Number of item groups
        iterations=500,    # Number of EM iterations
        sampling=5,        # Number of parallel runs
        seed=1            # For reproducibility
    )

4. Fit the model::

    model.fit(data)

5. Make predictions::

    predictions = model.predict(test_data)

   ``test_data`` needs only the user and item columns; with a rating column
   too, ``model.score()`` compares the predictions with it.

Complete Example
----------------

Here's a complete example with synthetic data::

    import pandas as pd
    import numpy as np
    from mmsbm import MMSBM

    # Generate synthetic data
    np.random.seed(42)
    n_users = 100
    n_items = 50

    train_data = pd.DataFrame({
        'users': [f'user{i}' for i in np.random.randint(0, n_users, 1000)],
        'items': [f'item{i}' for i in np.random.randint(0, n_items, 1000)],
        'ratings': np.random.randint(1, 6, 1000)
    })

    # Initialize and fit model
    model = MMSBM(user_groups=3, item_groups=5)
    model.fit(train_data)

    # Get model results
    results = model.score()
    print(f"Accuracy: {results['stats']['accuracy']:.3f}")
    print(f"MAE: {results['stats']['mae']:.3f}")

``accuracy``, ``one_off_accuracy`` and ``mae`` evaluate the most probable
rating; ``expected_rating_accuracy`` and ``expected_rating_mae`` evaluate the
expected rating; ``likelihood`` is the log-likelihood of the training data.
Errors are in rating units when the rating labels are numbers.

Cross-Validation
----------------

To use cross-validation::

    accuracies = model.cv_fit(data, folds=5)
    print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

Each user's ratings are spread across the folds, so every user with at least
two ratings stays in training in every fold.

Model Parameters
----------------

Key parameters when initializing MMSBM:

* ``user_groups``: Number of user groups (K)
* ``item_groups``: Number of item groups (L)
* ``chunk_size``: Rows per block of the E-step; by default as many as keep the responsibility tensor under 256 MiB
* ``log_file``: File that receives a copy of the log messages (``None`` for console only)
* ``user_col``, ``item_col``, ``rating_col``: Names of the data columns (default ``users``, ``items``, ``ratings``)
* ``iterations``: Number of EM iterations
* ``sampling``: Number of parallel runs
* ``seed``: Random seed for reproducibility
* ``debug``: Enable debug logging
* ``backend``: Computational backend (``numpy``, ``numba``, ``cupy`` or ``auto``)
* ``degree_corrected``: Fit the degree-corrected variant (default ``False``)

Degree-corrected model
----------------------

The standard model only explains the ratings of the pairs that are observed.
The degree-corrected variant also explains *which* pairs are rated, with the
degrees of users and items absorbed by the parameters instead of driving the
groups (see :doc:`../theory/mmsbm`). It is selected with a flag and used in
exactly the same way::

    model = MMSBM(user_groups=3, item_groups=4, iterations=500, sampling=5,
                  degree_corrected=True)
    model.fit(train_data)
    predictions = model.predict(test_data)     # p(r | u, i), as in the standard model
    results = model.score()

    # Probability that each (user, item) pair is rated at all
    link_probabilities = model.predict_links(test_data[["users", "items"]])

With the flag on, the rows of ``theta`` and ``eta`` sum to the number of
ratings of each user and item; divide a row by its sum to get the membership
distribution. With the flag off, results are identical to previous versions.

Choosing the number of groups
-----------------------------

``select_groups`` fits every combination of a grid of user and item group
numbers with k-fold cross-validation and scores it by the mean held-out
log-likelihood of the ratings::

    model = MMSBM(user_groups=2, item_groups=2, iterations=300, sampling=3, seed=1)
    grid = model.select_groups(data, user_groups=[2, 3, 4], item_groups=[2, 4, 6], folds=5)
    print(grid.sort_values("log_likelihood", ascending=False))

    best = MMSBM(*model.selected_groups, iterations=300, sampling=3)
    best.fit(data)

The grid costs ``len(user_groups) * len(item_groups) * folds`` fits. Predicted
probabilities are smoothed towards the uniform distribution before taking logs
(``smoothing``, default 0.05), and ``criterion="accuracy"`` selects by held-out
accuracy instead. The same log-likelihood is available for a single fit, after
``predict``, through ``model.held_out_log_likelihood()``.

Additional Features
-------------------

Parallel Processing
^^^^^^^^^^^^^^^^^^^

MMSBM automatically uses parallel processing for different sampling runs::

    # Will use 10 parallel processes
    model = MMSBM(sampling=10)
    model.fit(data)

Debug Mode
^^^^^^^^^^

For debugging and monitoring::

    model = MMSBM(debug=True)
    model.fit(data)

This will print additional information during fitting.

Next Steps
----------

* Check the :doc:`../theory/mmsbm` for theoretical background
* See :doc:`../theory/em_algorithm` for optimization details
* Explore the :doc:`../api/modules` for detailed API reference
