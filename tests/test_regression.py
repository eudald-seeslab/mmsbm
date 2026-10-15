"""With ``degree_corrected=False`` the model must reproduce the results of
version 1.0.7, before the degree-corrected variant existed.

``tests/data/regression_numpy_v1_0_7.npz`` was generated with the 1.0.7 code
and the numpy backend by fitting the models in :func:`compute` and storing,
for each, the prediction matrix, the exposed parameters and the stats
(``accuracy, one_off_accuracy, mae, s2, s2pond, likelihood`` in this order),
plus the parameters and likelihoods of every run of the model with two
sampling runs and the fold accuracies of a ``cv_fit``.  The tolerance is far
tighter than the 1% used elsewhere in the suite: any change to the standard
code path shows up here.

Version 1.2.0 changed the definition of some of the stored quantities without
touching the EM, so they are compared through the 1.0.7 definition or not at
all:

- ``mae`` is now a mean absolute error; the 1.0.7 value is one minus the
  current ``expected_rating_accuracy``.
- ``likelihood`` is now the log-likelihood; the 1.0.7 values (a weighted
  entropy of the responsibilities) are not compared.
- ``cv_fit`` splits the data differently; the stored fold accuracies are not
  compared.
- With ``sampling > 1`` the exposed run is now the one with the highest
  training log-likelihood instead of the highest test accuracy, so the run is
  selected here the 1.0.7 way before comparing the exposed parameters.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from mmsbm import MMSBM

REFERENCE = Path(__file__).parent / "data" / "regression_numpy_v1_0_7.npz"
NOT_COMPARED = {"big_run_likelihoods", "cv_acc"}


def mock_data(seed, n, n_users=5, n_items=10):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "users": [f"user{rng.choice(list(range(n_users)))}" for _ in range(n)],
            "items": [f"item{rng.choice(list(range(n_items)))}" for _ in range(n)],
            "ratings": [rng.choice(list(range(1, 6))) for _ in range(n)],
        }
    )


def stats_as_in_1_0_7(stats):
    """The first five entries of the 1.0.7 stats vector, from the 1.2.0 stats."""
    return np.array(
        [
            stats["accuracy"],
            stats["one_off_accuracy"],
            1 - stats["expected_rating_accuracy"],
            stats["s2"],
            stats["s2pond"],
        ]
    )


def best_run_as_in_1_0_7(model):
    """Index of the run with the highest accuracy on the data passed to predict."""
    rats = [model.em.compute_prod_dist(model.test, r["theta"], r["eta"], r["pr"]) for r in model.results]
    accuracies = [model._compute_stats(a)["accuracy"] for a in rats]
    return accuracies.index(max(accuracies))


def objects_and_stats(model, prefix):
    run = model.results[best_run_as_in_1_0_7(model)]
    handler = model.data_handler
    pr = handler.return_pr_indices(run["pr"])
    return {
        f"{prefix}_pred": model.prediction_matrix,
        f"{prefix}_theta": handler.return_theta_indices(run["theta"]).values,
        f"{prefix}_eta": handler.return_eta_indices(run["eta"]).values,
        f"{prefix}_pr": np.stack([v.values for v in pr.values()], axis=-1),
        f"{prefix}_stats": stats_as_in_1_0_7(model.score(silent=True)["stats"]),
    }


def compute(backend="numpy"):
    store = {}

    model = MMSBM(2, 2, iterations=10, seed=1, backend=backend)
    model.fit(mock_data(1, 100), silent=True)
    model.predict(mock_data(2, 100))
    store.update(objects_and_stats(model, "small"))

    model = MMSBM(3, 4, iterations=30, sampling=2, seed=7, backend=backend)
    model.fit(mock_data(3, 3000, 60, 80), silent=True)
    model.predict(mock_data(4, 500, 60, 80))
    store.update(objects_and_stats(model, "big"))
    store["big_run_thetas"] = np.stack([r["theta"] for r in model.results])

    return store


def test_standard_model_matches_v1_0_7_reference():
    reference = np.load(REFERENCE)
    computed = compute()
    assert set(computed) | NOT_COMPARED == set(reference.files)
    for key, value in computed.items():
        expected = reference[key][: len(value)] if key.endswith("_stats") else reference[key]
        np.testing.assert_allclose(value, expected, rtol=1e-8, atol=1e-12, err_msg=key)
