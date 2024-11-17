Quick Start Guide
==============

This guide will help you get started with MMSBM quickly.

Basic Usage
---------

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

Complete Example
--------------

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

Cross-Validation
--------------

To use cross-validation::

    accuracies = model.cv_fit(data, folds=5)
    print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

Model Parameters
-------------

Key parameters when initializing MMSBM:

* ``user_groups``: Number of user groups (K)
* ``item_groups``: Number of item groups (L)
* ``iterations``: Number of EM iterations
* ``sampling``: Number of parallel runs
* ``seed``: Random seed for reproducibility
* ``debug``: Enable debug logging

Additional Features
----------------

Parallel Processing
^^^^^^^^^^^^^^^^

MMSBM automatically uses parallel processing for different sampling runs::

    # Will use 10 parallel processes
    model = MMSBM(sampling=10)
    model.fit(data)

Debug Mode
^^^^^^^^^

For debugging and monitoring::

    model = MMSBM(debug=True)
    model.fit(data)

This will print additional information during fitting.

Next Steps
---------

* Check the :doc:`../theory/mmsbm` for theoretical background
* See :doc:`../theory/em_algorithm` for optimization details
* Explore the :doc:`../api/modules` for detailed API reference
