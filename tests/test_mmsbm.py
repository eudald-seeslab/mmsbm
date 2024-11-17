import pytest
import pandas as pd
import numpy as np

from mmsbm import MMSBM


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
    mm = MMSBM(2, 2, seed=1)
    mm.fit(mock_data(1))

    return mm


def cv_fit_model():
    mm = MMSBM(2, 2, seed=1)

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

    assert accuracies[0] == pytest.approx(0.08, 1)
    assert accuracies[1] == pytest.approx(0.12, 1)


class TestStats:
    def test_accuracy(self, check_score):
        assert check_score["stats"]["accuracy"] == pytest.approx(0.24, 0.01)

    def test_one_off_accuracy(self, check_score):
        assert check_score["stats"]["one_off_accuracy"] == pytest.approx(0.47, 0.01)

    def test_mae(self, check_score):
        assert check_score["stats"]["mae"] == pytest.approx(0.78, 0.01)

    def test_s2(self, check_score):
        assert check_score["stats"]["s2"] == pytest.approx(179, 1)

    def test_s2pond(self, check_score):
        assert check_score["stats"]["s2"] == pytest.approx(163, 0.01)

    def test_likelihood(self, check_score):
        assert check_score["stats"]["likelihood"].sum() == pytest.approx(-137, 1)


class TestObjects:
    def test_theta(self, check_score):
        assert check_score["objects"]["theta"].sum(axis=0)[0] == pytest.approx(
            1.35, 0.01
        )

    def test_eta(self, check_score):
        assert check_score["objects"]["eta"].sum(axis=0)[0] == pytest.approx(3.74, 0.01)

    def test_pr_keys(self, check_score):
        assert set(list(check_score["objects"]["pr"].keys())) == {"1", "2", "3", "4", "5"}

    def test_pr_values(self, check_score):
        correct_values = [0.67, 0.65, 0.88, 1.05, 0.75]

        assert [
            a.sum().sum() == pytest.approx(b, 0.01)
            for (a, b) in zip(check_score["objects"]["pr"].values(), correct_values)
        ]
