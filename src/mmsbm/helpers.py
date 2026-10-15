import numpy as np


def _invert_dict(d):
    return {v: k for k, v in d.items()}


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
