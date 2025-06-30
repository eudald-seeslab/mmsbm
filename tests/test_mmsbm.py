import pytest
import pandas as pd
import numpy as np

from mmsbm import MMSBM
from data_handler import DataHandler


RATING_NUM = 100


def mock_data(seed, n=RATING_NUM):

    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "users": [f"user{rng.choice(list(range(5)))}" for _ in range(n)],
            "items": [f"item{rng.choice(list(range(10)))}" for _ in range(n)],
            "ratings": [rng.choice(list(range(1, 6))) for _ in range(n)],
        }
    )


def fit_model():
    mm = MMSBM(2, 2, iterations=10, seed=1)
    mm.fit(mock_data(1))

    return mm


def cv_fit_model():
    mm = MMSBM(2, 2, iterations=10, seed=1)

    return mm.cv_fit(mock_data(1), folds=2)


@pytest.fixture
def fit_and_predict():
    mm = fit_model()
    pred_matrix = mm.predict(mock_data(2))

    return pred_matrix


@pytest.fixture
def check_score():
    mm = fit_model()
    _ = mm.predict(mock_data(2))
    return mm.score(silent=True)


def test_prediction_matrix(fit_and_predict):
    assert fit_and_predict.sum() == pytest.approx(RATING_NUM, 0.01)


def test_cv_fit():
    accuracies = cv_fit_model()

    assert accuracies[0] == pytest.approx(0.125, 0.01)
    assert accuracies[1] == pytest.approx(0.16, 0.01)


class TestStats:
    def test_accuracy(self, check_score):
        assert check_score["stats"]["accuracy"] == pytest.approx(0.13, 0.01)

    def test_one_off_accuracy(self, check_score):
        assert check_score["stats"]["one_off_accuracy"] == pytest.approx(0.55, 0.01)

    def test_mae(self, check_score):
        assert check_score["stats"]["mae"] == pytest.approx(0.78, 0.01)

    def test_s2(self, check_score):
        assert check_score["stats"]["s2"] == pytest.approx(153, 1)

    def test_s2pond(self, check_score):
        assert check_score["stats"]["s2pond"] == pytest.approx(129.48, 1)

    def test_likelihood(self, check_score):
        assert check_score["stats"]["likelihood"].sum() == pytest.approx(-13.77, 1)


class TestObjects:
    def test_theta(self, check_score):
        assert check_score["objects"]["theta"].sum(axis=0)[0] == pytest.approx(
            2.11, 0.1
        )

    def test_eta(self, check_score):
        assert check_score["objects"]["eta"].sum(axis=0)[0] == pytest.approx(5.93, 0.1)

    def test_pr_keys(self, check_score):
        assert set(list(check_score["objects"]["pr"].keys())) == {"1", "2", "3", "4", "5"}

    def test_pr_values(self, check_score):
        correct_values = [0.79, 0.85, 0.93, 0.71, 0.72]

        assert [
            a.sum().sum() == pytest.approx(b, 0.01)
            for (a, b) in zip(check_score["objects"]["pr"].values(), correct_values)
        ]


# ---------------------------------------------------------------------------
# Additional edge-case and internal-helper tests (merged from test_mmsbm_extra)
# ---------------------------------------------------------------------------


def test_predict_without_fit_raises():
    """Calling predict before fit should raise an AssertionError."""
    mm = MMSBM(2, 2, seed=1)
    with pytest.raises(AssertionError):
        _ = mm.predict(mock_data(0))


def test_score_without_predict_raises():
    """Calling score before predict should raise an AssertionError."""
    mm = MMSBM(2, 2, seed=1)
    with pytest.raises(AssertionError):
        _ = mm.score()


def test_choose_best_run_selects_highest_accuracy(monkeypatch):
    """The helper should return the index with the highest computed accuracy."""
    mm = MMSBM(1, 1, seed=1)

    # Monkey-patch `_compute_stats` so that its accuracy is simply the input value
    monkeypatch.setattr(mm, "_compute_stats", lambda x: {"accuracy": x})

    accuracies = [0.1, 0.7, 0.3]
    assert mm.choose_best_run(accuracies) == 1  # index of 0.7


def test_compute_likelihood_is_finite():
    """The optimised likelihood computation should always return a finite value."""
    data_df = mock_data(1, n=10)

    mm = MMSBM(2, 2, iterations=1, sampling=1, seed=1, backend="numpy")

    # Manually prepare internal objects without spawning subprocesses
    dh = DataHandler()
    train = dh.format_train_data(data_df)
    mm._prepare_objects(train)

    K = mm._dims["n_user_groups"]
    L = mm._dims["n_item_groups"]
    R = mm._dims["n_ratings"]

    rng = np.random.default_rng(0)

    theta = rng.random((mm.p + 1, K))
    theta /= theta.sum(axis=1, keepdims=True)

    eta = rng.random((mm.m + 1, L))
    eta /= eta.sum(axis=1, keepdims=True)

    pr = rng.random((K, L, R))
    pr /= pr.sum(axis=2, keepdims=True)

    ll_value = mm.compute_likelihood(mm.train, theta, eta, pr)

    assert np.isfinite(ll_value)


def test_run_one_sampling_executes_and_returns_expected_keys():
    """Running a single EM sampling in-process should return expected keys."""
    data_df = mock_data(2, n=15)

    mm = MMSBM(2, 2, iterations=1, sampling=1, seed=2, backend="numpy")

    dh = DataHandler()
    train = dh.format_train_data(data_df)
    mm._prepare_objects(train)

    result = mm.run_one_sampling(train, seed=123, i=0)

    assert {"likelihood", "pr", "theta", "eta"} == set(result.keys())
    assert result["pr"].shape[:2] == (
        mm._dims["n_user_groups"],
        mm._dims["n_item_groups"],
    )


def test_score_with_logging():
    """Ensure the logging branch (`silent=False`) executes without errors."""
    mm = MMSBM(2, 2, iterations=1, sampling=1, seed=3, backend="numpy")
    mm.fit(mock_data(3, n=20), silent=True)
    _ = mm.predict(mock_data(4, n=10))
    res = mm.score(silent=False)

    assert {"stats", "objects"} == set(res.keys())
    assert "accuracy" in res["stats"]
