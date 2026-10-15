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

    assert accuracies[0] == pytest.approx(0.294, 0.01)
    assert accuracies[1] == pytest.approx(0.265, 0.01)


def test_cv_fit_sparse_data_uses_every_fold():
    """Users with far fewer ratings than items / folds: every fold must still hold out about a fifth of the ratings and train on the rest (before 1.2.0 the first fold took everything)."""
    rng = np.random.default_rng(0)
    n_users, n_items, per_user = 300, 1000, 20
    data = pd.DataFrame(
        {
            "users": np.repeat([f"user{u}" for u in range(n_users)], per_user),
            "items": [f"item{i}" for i in rng.integers(n_items, size=n_users * per_user)],
            "ratings": rng.integers(1, 6, size=n_users * per_user),
        }
    )
    mm = MMSBM(2, 2, iterations=2, seed=1)

    accuracies = mm.cv_fit(data, folds=5)

    assert len(accuracies) == 5
    assert all(0 <= a <= 1 for a in accuracies)
    # Last fold: trained on 4 / 5 of the ratings, tested on the rest (minus the few whose item was unseen)
    assert len(mm.train) == n_users * per_user * 4 // 5
    assert 1000 <= len(mm.test) <= 1200


def test_cv_fit_rejects_bad_fold_numbers():
    mm = MMSBM(2, 2, iterations=1, seed=1)
    with pytest.raises(AssertionError):
        mm.cv_fit(mock_data(1), folds=1)
    with pytest.raises(AssertionError):
        mm.cv_fit(mock_data(1), folds=RATING_NUM + 1)


class TestStats:
    def test_accuracy(self, check_score):
        assert check_score["stats"]["accuracy"] == pytest.approx(0.13, 0.01)

    def test_one_off_accuracy(self, check_score):
        assert check_score["stats"]["one_off_accuracy"] == pytest.approx(0.55, 0.01)

    def test_mae(self, check_score):
        assert check_score["stats"]["mae"] == pytest.approx(1.53, 0.01)

    def test_expected_rating_accuracy(self, check_score):
        assert check_score["stats"]["expected_rating_accuracy"] == pytest.approx(0.22, 0.01)

    def test_expected_rating_mae(self, check_score):
        assert check_score["stats"]["expected_rating_mae"] == pytest.approx(1.2948, 0.01)

    def test_s2(self, check_score):
        assert check_score["stats"]["s2"] == pytest.approx(153, 1)

    def test_s2pond(self, check_score):
        assert check_score["stats"]["s2pond"] == pytest.approx(129.48, 1)

    def test_sums_are_means_times_n(self, check_score):
        stats = check_score["stats"]
        assert stats["s2"] == pytest.approx(stats["mae"] * RATING_NUM)
        assert stats["s2pond"] == pytest.approx(stats["expected_rating_mae"] * RATING_NUM)

    def test_likelihood(self, check_score):
        assert check_score["stats"]["likelihood"] == pytest.approx(-147.96, 0.01)


def stats_for(labels, rat, real):
    """Stats of a hand-built prediction matrix ``rat`` for ratings ``real`` (indices into ``labels``)."""
    mm = MMSBM(1, 1, seed=1)
    mm.data_handler = DataHandler()
    mm.data_handler.ratings_dict = {label: i for i, label in enumerate(labels)}
    real = np.asarray(real)
    mm.test = np.column_stack([np.zeros(len(real), int), np.zeros(len(real), int), real])
    return mm._compute_stats(np.asarray(rat, dtype=float))


class TestStatsDefinitions:
    def test_integer_ratings(self):
        stats = stats_for(
            ["1", "2", "3", "4", "5"],
            [[0.7, 0.3, 0, 0, 0],   # most probable 1, expected 1.3 -> nearest 1; real 1
             [0, 0, 0.1, 0.9, 0],   # most probable 4, expected 3.9 -> nearest 4; real 2
             [0.5, 0.5, 0, 0, 0]],  # tie: most probable 1, expected 1.5 -> nearest (lower) 1; real 2
            [0, 1, 1],
        )
        assert stats["accuracy"] == pytest.approx(1 / 3)
        assert stats["one_off_accuracy"] == pytest.approx(2 / 3)
        assert stats["mae"] == pytest.approx((0 + 2 + 1) / 3)
        assert stats["expected_rating_accuracy"] == pytest.approx(1 / 3)
        assert stats["expected_rating_mae"] == pytest.approx((0.3 + 1.9 + 0.5) / 3)
        assert stats["s2"] == pytest.approx(3)
        assert stats["s2pond"] == pytest.approx(2.7)

    def test_errors_are_in_rating_units(self):
        """Half-star ratings: two levels apart is one unit apart."""
        stats = stats_for(
            ["0.5", "1.0", "1.5", "2.0"],
            [[0.7, 0.3, 0, 0], [0, 0, 0.1, 0.9]],
            [0, 1],
        )
        assert stats["accuracy"] == pytest.approx(0.5)
        assert stats["one_off_accuracy"] == pytest.approx(1.0)
        assert stats["mae"] == pytest.approx(0.5)
        assert stats["expected_rating_accuracy"] == pytest.approx(0.5)
        assert stats["expected_rating_mae"] == pytest.approx((0.15 + 0.95) / 2)

    def test_ratings_sorted_as_strings_still_measured_by_value(self):
        """The data handler sorts labels as strings ('1', '10', '2', ...); errors must use the values."""
        labels = DataHandler._create_values_dict([str(r) for r in range(1, 11)])
        labels = [label for label, _ in sorted(labels.items(), key=lambda kv: kv[1])]
        assert labels[:3] == ["1", "10", "2"]
        one_hot = np.eye(10)
        stats = stats_for(labels, [one_hot[1]], [2])  # predicts "10", real "2"
        assert stats["mae"] == pytest.approx(8)
        assert stats["one_off_accuracy"] == 0

    def test_non_numeric_ratings_fall_back_to_positions(self):
        stats = stats_for(["bad", "good", "great"], [[0, 0, 1], [0, 1, 0]], [0, 0])
        assert stats["mae"] == pytest.approx(1.5)
        assert stats["one_off_accuracy"] == pytest.approx(0.5)

    def test_all_zero_rows_are_left_out(self):
        stats = stats_for(["1", "2"], [[0, 0], [1, 0]], [1, 0])
        assert stats["accuracy"] == 1
        assert stats["s2"] == 0


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


def test_choose_best_run_selects_highest_likelihood():
    """The run exposed after predict is the one with the highest training log-likelihood (first on ties)."""
    mm = MMSBM(1, 1, seed=1)
    mm.results = [{"likelihood": -10.0}, {"likelihood": -3.0}, {"likelihood": -3.0}, {"likelihood": -7.0}]

    assert mm.choose_best_run() == 1
    assert mm.choose_best_run([0.9, 0.1, 0.1, 0.1]) == 1  # the old argument is ignored


def test_choose_best_run_requires_fit():
    with pytest.raises(AssertionError):
        MMSBM(1, 1, seed=1).choose_best_run()


def test_predict_exposes_run_with_highest_likelihood():
    mm = MMSBM(2, 2, iterations=10, sampling=2, seed=1, backend="numpy")
    mm.fit(mock_data(1), silent=True)
    mm.predict(mock_data(2))

    likelihoods = [float(a["likelihood"]) for a in mm.results]
    best = int(np.argmax(likelihoods))
    assert likelihoods[0] != likelihoods[1]
    assert mm.likelihood == likelihoods[best]
    np.testing.assert_array_equal(mm.theta.values, mm.results[best]["theta"])
    np.testing.assert_array_equal(mm.eta.values, mm.results[best]["eta"])


def test_compute_likelihood_is_log_probability_of_the_ratings():
    """likelihood = Σ log p(r | u, i), the probability of the observed rating under the fitted model."""
    mm = fit_model()
    run = mm.results[0]

    probabilities = mm.em.compute_prod_dist(mm.train, run["theta"], run["eta"], run["pr"])
    manual = np.sum(np.log(probabilities[np.arange(len(mm.train)), mm.train[:, 2]]))

    assert run["likelihood"] == pytest.approx(manual)
    assert mm.compute_likelihood(mm.train, run["theta"], run["eta"], run["pr"]) == pytest.approx(manual)
    assert run["likelihood"] <= 0


def test_em_never_decreases_the_likelihood():
    train = DataHandler().format_train_data(mock_data(3, n=3000))
    mm = MMSBM(3, 4, iterations=1, seed=0, backend="numpy")
    mm._prepare_objects(train)
    rng = np.random.default_rng(0)
    theta = mm.em.normalize_with_d(rng.random((mm.p + 1, 3)), "user")
    eta = mm.em.normalize_with_d(rng.random((mm.m + 1, 4)), "item")
    pr = mm.em.normalize_with_self(rng.random((3, 4, 5)))

    history = []
    for _ in range(25):
        n_theta, n_eta, n_pr = mm.em.update_coefficients(train, theta, eta, pr)
        theta = mm.em.normalize_with_d(n_theta, "user")
        eta = mm.em.normalize_with_d(n_eta, "item")
        pr = mm.em.normalize_with_self(n_pr)
        history.append(mm.em.compute_likelihood(train, theta, eta, pr))

    assert np.all(np.diff(history) >= -1e-9)
    assert history[-1] > history[0]


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


def test_log_file_none_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    MMSBM(2, 2, iterations=1, seed=1, log_file=None).fit(mock_data(1), silent=False)
    assert not (tmp_path / "mmsbm.log").exists()

    MMSBM(2, 2, iterations=1, seed=1).fit(mock_data(1), silent=False)
    assert (tmp_path / "mmsbm.log").exists()


def test_chunk_size_does_not_change_the_fit():
    results = {}
    for chunk_size in (None, 7):
        mm = MMSBM(2, 2, iterations=10, seed=1, backend="numpy", chunk_size=chunk_size)
        mm.fit(mock_data(1), silent=True)
        results[chunk_size] = (mm.predict(mock_data(2)), mm.results[0]["likelihood"])
    np.testing.assert_allclose(results[7][0], results[None][0], rtol=1e-10)
    assert results[7][1] == pytest.approx(results[None][1])


def test_score_with_logging():
    """Ensure the logging branch (`silent=False`) executes without errors."""
    mm = MMSBM(2, 2, iterations=1, sampling=1, seed=3, backend="numpy")
    mm.fit(mock_data(3, n=20), silent=True)
    _ = mm.predict(mock_data(4, n=10))
    res = mm.score(silent=False)

    assert {"stats", "objects"} == set(res.keys())
    assert "accuracy" in res["stats"]
