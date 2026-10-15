import numpy as np
import pandas as pd
import pytest

from helpers import assign_folds, drop_unseen
from mmsbm import MMSBM
from tests.synthetic import planted_data

GRID_COLUMNS = [
    "user_groups", "item_groups", "log_likelihood", "log_likelihood_sd", "accuracy", "accuracy_sd",
]


def test_assign_folds_spreads_each_users_ratings():
    data, _, _, _ = planted_data(3, n_users=50, n_items=80, density=0.1)
    fold_ids = assign_folds(data, 5, np.random.default_rng(0))
    assert fold_ids.shape == (len(data),)
    assert set(fold_ids) <= set(range(5))
    counts = pd.crosstab(data["users"].to_numpy(), fold_ids)
    assert (counts.max(axis=1) - counts.min(axis=1) <= 1).all()


def test_assign_folds_is_reproducible():
    data, _, _, _ = planted_data(3, n_users=20, n_items=30, density=0.2)
    first = assign_folds(data, 3, np.random.default_rng(5))
    second = assign_folds(data, 3, np.random.default_rng(5))
    assert np.array_equal(first, second)


def test_drop_unseen():
    train = pd.DataFrame({"users": ["a", "b"], "items": ["x", "y"], "ratings": [1, 2]})
    test = pd.DataFrame({"users": ["a", "c", "b"], "items": ["y", "x", "z"], "ratings": [1, 1, 1]})
    assert drop_unseen(train, test).index.tolist() == [0]


def test_select_groups_finds_planted_sizes_with_degree_correction():
    """Planted 2 x 2 structure in both links and ratings: the degree-corrected
    model sees both and the cross-validated log-likelihood must pick (2, 2)."""
    data, _, _, _ = planted_data(0, n_users=200, n_items=300, density=0.1, informative=True)
    model = MMSBM(2, 2, iterations=80, sampling=2, seed=1, backend="numpy", degree_corrected=True)
    grid = model.select_groups(data, user_groups=[1, 2], item_groups=[1, 2], folds=2)

    assert list(grid.columns) == GRID_COLUMNS
    assert len(grid) == 4
    assert (grid["log_likelihood"] <= 0).all()
    assert model.selected_groups == (2, 2)
    scores = grid.set_index(["user_groups", "item_groups"])
    assert scores.loc[(2, 2), "log_likelihood"] > scores.loc[(1, 1), "log_likelihood"] + 0.1
    assert scores.loc[(2, 2), "accuracy"] > scores.loc[(1, 1), "accuracy"]


@pytest.mark.parametrize("criterion", ["log_likelihood", "accuracy"])
def test_select_groups_is_consistent_with_its_criterion(criterion):
    data, _, _, _ = planted_data(0, n_users=100, n_items=150, density=0.2, informative=True)
    model = MMSBM(2, 2, iterations=30, sampling=1, seed=1, backend="numpy")
    grid = model.select_groups(data, user_groups=[1, 2], item_groups=[2], folds=2, criterion=criterion)

    assert list(grid.columns) == GRID_COLUMNS
    assert len(grid) == 2
    best = grid.loc[grid[criterion].idxmax()]
    assert model.selected_groups == (int(best["user_groups"]), int(best["item_groups"]))


def test_select_groups_rejects_unknown_criterion():
    data, _, _, _ = planted_data(0, n_users=20, n_items=30, density=0.3)
    model = MMSBM(2, 2, iterations=5, seed=1, backend="numpy")
    with pytest.raises(AssertionError):
        model.select_groups(data, user_groups=[1], item_groups=[1], folds=2, criterion="mae")


def test_held_out_log_likelihood_smoothing_bounds_the_penalty():
    data, _, _, _ = planted_data(0, n_users=40, n_items=60, density=0.3, informative=True)
    model = MMSBM(2, 2, iterations=30, seed=1, backend="numpy")
    model.fit(data, silent=True)
    model.predict(data)
    raw = model.held_out_log_likelihood()
    smoothed = model.held_out_log_likelihood(smoothing=0.05)
    n_ratings = model.prediction_matrix.shape[1]
    assert smoothed >= np.log(0.05 / n_ratings)
    assert raw <= 0 and smoothed <= 0
    with pytest.raises(AssertionError):
        model.held_out_log_likelihood(smoothing=2)
