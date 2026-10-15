import numpy as np
import pandas as pd
import pytest

from mmsbm.backend import load_backend
from mmsbm.data_handler import DataHandler
from mmsbm.expectation_maximization import ExpectationMaximization
from mmsbm import MMSBM, MMSBMError
from tests.synthetic import planted_data, best_agreement, ordered


def mock_data(seed, n=100):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "users": [f"user{rng.choice(list(range(5)))}" for _ in range(n)],
            "items": [f"item{rng.choice(list(range(10)))}" for _ in range(n)],
            "ratings": [rng.choice(list(range(1, 6))) for _ in range(n)],
        }
    )


@pytest.fixture(scope="module")
def planted():
    """Uninformative ratings: the groups are only visible in which pairs are rated."""
    return planted_data(0)


@pytest.fixture(scope="module")
def dc_model(planted):
    data = planted[0]
    model = MMSBM(2, 2, iterations=80, sampling=2, seed=1, backend="numpy", degree_corrected=True)
    model.fit(data, silent=True)
    model.predict(data)
    return model


def test_memberships_sum_to_degrees(dc_model, planted):
    data = planted[0]
    objects = dc_model.score(silent=True)["objects"]
    user_degrees = data.groupby("users").size()
    item_degrees = data.groupby("items").size()
    np.testing.assert_allclose(
        objects["theta"].sum(axis=1).loc[list(user_degrees.index)].to_numpy(), user_degrees.to_numpy()
    )
    np.testing.assert_allclose(
        objects["eta"].sum(axis=1).loc[list(item_degrees.index)].to_numpy(), item_degrees.to_numpy()
    )


def test_expected_number_of_ratings_equals_observed(dc_model, planted):
    for run in dc_model.results:
        expected = np.einsum("k,l,klr->", run["theta"].sum(axis=0), run["eta"].sum(axis=0), run["pr"])
        assert expected == pytest.approx(len(planted[0]))


def test_prediction_rows_are_distributions(dc_model):
    np.testing.assert_allclose(dc_model.prediction_matrix.sum(axis=1), 1.0)


def test_recovers_link_structure_without_rating_signal(dc_model, planted):
    _, g, h, _ = planted
    objects = dc_model.score(silent=True)["objects"]
    assert best_agreement(ordered(objects["theta"], "u", len(g)), g, 2) > 0.9
    assert best_agreement(ordered(objects["eta"], "i", len(h)), h, 2) > 0.9


def test_groups_are_not_driven_by_degree(dc_model, planted):
    _, g, _, degree_factor = planted
    theta = ordered(dc_model.score(silent=True)["objects"]["theta"], "u", len(g))
    heavy = (degree_factor > 1).astype(int)
    assert best_agreement(theta, heavy, 2) < 0.65


def test_standard_model_cannot_see_link_structure(planted):
    data, g, h, _ = planted
    model = MMSBM(2, 2, iterations=80, sampling=2, seed=1, backend="numpy", degree_corrected=False)
    model.fit(data, silent=True)
    model.predict(data)
    objects = model.score(silent=True)["objects"]
    assert best_agreement(ordered(objects["theta"], "u", len(g)), g, 2) < 0.75
    assert best_agreement(ordered(objects["eta"], "i", len(h)), h, 2) < 0.75


def test_predict_links_reflects_affinity_and_degree(dc_model, planted):
    _, g, h, degree_factor = planted
    n_users, n_items = len(g), len(h)
    pairs = pd.DataFrame(
        {
            "users": [f"u{a}" for a in range(n_users) for _ in range(n_items)],
            "items": [f"i{b}" for _ in range(n_users) for b in range(n_items)],
        }
    )
    probabilities = dc_model.predict_links(pairs).reshape(n_users, n_items)
    assert probabilities.min() >= 0 and probabilities.max() <= 1
    same_group = g[:, None] == h[None, :]
    assert probabilities[same_group].mean() > 2 * probabilities[~same_group].mean()
    assert probabilities[degree_factor > 1].mean() > 2 * probabilities[degree_factor == 1].mean()


def test_predict_links_ignores_ratings_and_drops_unseen(dc_model, planted):
    data = planted[0].head(10).copy()
    data.loc[data.index[0], "users"] = "unknown_user"
    assert dc_model.predict_links(data).shape == (9,)


def test_predict_links_requires_degree_corrected():
    model = MMSBM(2, 2, iterations=5, seed=1, backend="numpy")
    model.fit(mock_data(1), silent=True)
    with pytest.raises(MMSBMError):
        model.predict_links(mock_data(2))


def test_poisson_log_likelihood_never_decreases(planted):
    train = DataHandler().format_train_data(planted[0])
    model = MMSBM(2, 2, iterations=1, seed=0, backend="numpy", degree_corrected=True)
    model._prepare_objects(train)
    rng = np.random.default_rng(0)
    theta, eta, pr = rng.random((model.p + 1, 2)), rng.random((model.m + 1, 2)), rng.random((2, 2, 5))
    history = []
    for _ in range(50):
        n_theta, n_eta, n_pr = model.em.update_coefficients(train, theta, eta, pr)
        theta, eta, pr = model.em.m_step_degree_corrected(n_theta, n_eta, n_pr)
        history.append(model.em.poisson_log_likelihood(train, theta, eta, pr))
    history = np.array(history)
    assert np.all(np.diff(history) >= -1e-9 * np.abs(history[1:]))
    assert history[-1] > history[0]


def test_initial_affinity_has_full_contrast():
    rng = np.random.default_rng(0)
    for k, l in [(1, 1), (1, 3), (2, 2), (3, 2), (2, 5), (4, 4)]:
        pattern = ExpectationMaximization.initial_affinity(rng, k, l)
        assert pattern.shape == (k, l)
        # Every user group has exactly one preferred item group ...
        assert np.all((pattern == 10.0).sum(axis=1) == 1)
        # ... and the preferred item groups are spread as evenly as possible
        counts = (pattern == 10.0).sum(axis=0)
        assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("backend_name", ["numba", "cupy"])
def test_backends_agree_with_numpy(backend_name, planted):
    try:
        load_backend(backend_name)
    except Exception:  # pragma: no cover
        pytest.skip(f"{backend_name} backend not available in this environment")
    train = DataHandler().format_train_data(planted[0])
    runs = {}
    for backend in ("numpy", backend_name):
        model = MMSBM(2, 2, iterations=20, seed=0, backend=backend, degree_corrected=True)
        model._show_progress = False
        model._prepare_objects(train)
        runs[backend] = model.run_one_sampling(train, seed=123, i=0)
    for key in ("theta", "eta", "pr", "likelihood"):
        np.testing.assert_allclose(runs[backend_name][key], runs["numpy"][key], rtol=1e-6, atol=1e-10)


def test_held_out_log_likelihood_matches_manual(dc_model):
    real = dc_model.test[:, 2]
    manual = np.mean(np.log(dc_model.prediction_matrix[np.arange(len(real)), real]))
    assert dc_model.held_out_log_likelihood() == pytest.approx(manual)
    assert dc_model.held_out_log_likelihood() <= 0


def test_held_out_log_likelihood_requires_predictions():
    model = MMSBM(2, 2, seed=1)
    with pytest.raises(MMSBMError):
        model.held_out_log_likelihood()


def test_cv_fit_with_degree_corrected():
    model = MMSBM(2, 2, iterations=10, seed=1, backend="numpy", degree_corrected=True)
    accuracies = model.cv_fit(mock_data(1), folds=2)
    assert len(accuracies) == 2
    assert all(0 <= a <= 1 for a in accuracies)
