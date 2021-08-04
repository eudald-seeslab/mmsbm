import pytest
import pandas as pd
import numpy as np

from mmsbm import MMSBM


def mock_data(seed):

    np.random.seed(seed)
    return pd.DataFrame(
        {
            "users": [f"user{np.random.choice(list(range(5)))}" for _ in range(100)],
            "items": [f"item{np.random.choice(list(range(10)))}" for _ in range(100)],
            "ratings": [np.random.choice(list(range(1, 6))) for _ in range(100)]
        }
    )


def fit_model():
    mm = MMSBM(2, 2)
    mm.fit(mock_data(1))

    return mm


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
    pred_matrix = fit_and_predict
    assert pred_matrix.sum() == pytest.approx(100, 0.01)


class TestStats:
    def __init__(self, check_score):
        self.results = check_score

    def test_accuracy(self):
        assert self.results["stats"]["accuracy"] == pytest.approx(0.18, 0.01)

    def test_one_off_accuracy(self):
        assert self.results["stats"]["test_one_off_accuracy"] == pytest.approx(0.46, 0.01)

    def test_mae(self):
        assert self.results["stats"]["mae"] == pytest.approx(0.87, 0.01)

    def test_s2(self):
        assert self.results["stats"]["s2"] == pytest.approx(179, 1)

    def test_s2pond(self):
        assert self.results["stats"]["s2"] == pytest.approx(150.22, 0.01)

    def test_likelihood(self):
        assert self.results["stats"]["likelihood"].sum() == pytest.approx(-136.24, 0.01)


class TestObjects:
    def __init__(self, check_score):
        self.results = check_score

    def test_theta(self):
        assert self.results["objects"]["theta"].sum(axis=0) == pytest.approx([1.95, 3.05], 0.01)

    def test_eta(self):
        assert self.results["objects"]["eta"].sum(axis=0) == pytest.approx([5.52, 4.48], 0.01)

    def test_pr_keys(self):
        assert list(self.results["objects"]["pr"].keys()) == ['3', '4', '1', '5', '2']

    def test_pr_values(self):
        assert [a.sum().sum() for a in self.results["objects"]["pr"].values()] == pytest.approx(
            [0.67, 0.65, 0.88, 1.05, 0.75], 0.01
        )
