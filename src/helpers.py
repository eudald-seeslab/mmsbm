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


def assign_folds(data, folds, rng):
    """Assign every rating to a fold, spreading each user's ratings across folds.

    Rows are shuffled and then dealt round-robin *within each user* (starting
    from a random fold per user), so a user with ``n`` ratings appears in the
    test part of at most ``n`` folds and in the training part of every fold as
    soon as ``n >= 2``.  This keeps held-out evaluations comparable across
    folds and avoids dropping users that have nothing left in training.

    Parameters
    ----------
    data : DataFrame
        Ratings with the user column first.
    folds : int
        Number of folds.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of shape (len(data),)
        Fold id (0 to ``folds - 1``) of each row of ``data``.
    """
    n = len(data)
    order = rng.permutation(n)
    users = data.iloc[order, 0].astype(str).to_numpy()
    unique_users, inverse = np.unique(users, return_inverse=True)

    # Position of every row within its user's (shuffled) ratings
    by_user = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inverse))[:-1]]
    position = np.empty(n, dtype=int)
    position[by_user] = np.arange(n) - starts[inverse[by_user]]

    offsets = rng.integers(folds, size=len(unique_users))
    fold_ids = np.empty(n, dtype=int)
    fold_ids[order] = (position + offsets[inverse]) % folds
    return fold_ids


def drop_unseen(train, test):
    """Return the rows of ``test`` whose user and item both appear in ``train``."""
    keep = np.ones(len(test), dtype=bool)
    for col in (0, 1):
        seen = set(train.iloc[:, col].astype(str))
        keep &= test.iloc[:, col].astype(str).isin(seen).to_numpy()
    return test[keep]
