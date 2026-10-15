"""Synthetic data with planted group structure, shared by the tests.

Two user groups and two item groups (``n_groups`` in general), heterogeneous
user degrees (half the users rate ``heavy`` times more often) and an
assortative link structure: a user rates items of its own group ``affinity``
times more often than the others.  Ratings are uniform noise unless
``informative=True``, in which case same-group pairs rate 4-5 and cross-group
pairs rate 1-2 (with 20% noise).  With uninformative ratings the only signal
about the groups is *which* pairs are rated.
"""
from itertools import permutations

import numpy as np
import pandas as pd


def planted_data(seed, n_groups=2, n_users=200, n_items=300, density=0.08,
                 affinity=4.0, heavy=4.0, informative=False):
    rng = np.random.default_rng(seed)
    g = rng.integers(n_groups, size=n_users)
    h = rng.integers(n_groups, size=n_items)
    degree_factor = np.where(rng.random(n_users) < 0.5, 1.0, heavy)
    weight = degree_factor[:, None] * np.where(g[:, None] == h[None, :], affinity, 1.0)
    u, i = np.nonzero(rng.random(weight.shape) < weight * density / weight.mean())
    n = len(u)
    if informative:
        r = np.where(g[u] == h[i], rng.choice([4, 5], size=n), rng.choice([1, 2], size=n))
        r = np.where(rng.random(n) < 0.2, rng.integers(1, 6, size=n), r)
    else:
        r = rng.integers(1, 6, size=n)
    data = pd.DataFrame(
        {"users": [f"u{a}" for a in u], "items": [f"i{a}" for a in i], "ratings": r}
    )
    return data, g, h, degree_factor


def best_agreement(memberships, planted, n_groups):
    """Fraction of nodes whose dominant group is the planted one, up to relabelling."""
    labels = memberships.argmax(axis=1)
    return max(
        np.mean(np.asarray(perm)[labels] == planted) for perm in permutations(range(n_groups))
    )


def ordered(frame, prefix, n):
    """Rows of a theta/eta DataFrame in planted (u0, u1, ...) order, as an array."""
    return frame.loc[[f"{prefix}{a}" for a in range(n)]].to_numpy()
