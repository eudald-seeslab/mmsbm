import numpy as np


def _invert_dict(d):
    return {v: k for k, v in d.items()}


def get_n_per_group(x, n, rng):
    for i in reversed(range(n)):
        try:
            return rng.choice(x.index, i + 1, replace=False)
        except ValueError:
            pass


def structure_folds(data, folds):
    # How many different items we have?
    n_items = len(set(data.iloc[:, 1]))
    # Check that we haven't asked for too many folds
    assert (
        folds <= n_items
    ), f"Fold number can't be higher than {n_items} since this is the number of different items you have."
    # Return the number of items that we can have in each fold
    return int(n_items / folds)
