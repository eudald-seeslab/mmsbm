"""Agreement of the numba and cupy kernels with the numpy reference, GPU
residency of the cupy kernels, and the blocked (chunked) E-step.

Backends that cannot be loaded in this environment are skipped.
"""
import numpy as np
import pytest

from backend import load_backend
from expectation_maximization import ExpectationMaximization, CHUNK_BUDGET
from data_handler import DataHandler
from mmsbm import MMSBM


def random_problem(seed, n=5000, n_users=50, n_items=60, K=3, L=4, R=5):
    rng = np.random.default_rng(seed)
    data = np.column_stack(
        [rng.integers(n_users, size=n), rng.integers(n_items, size=n), rng.integers(R, size=n)]
    ).astype(np.int64)
    theta = rng.random((n_users, K))
    theta /= theta.sum(axis=1, keepdims=True)
    eta = rng.random((n_items, L))
    eta /= eta.sum(axis=1, keepdims=True)
    pr = rng.random((K, L, R))
    pr /= pr.sum(axis=2, keepdims=True)
    return data, theta, eta, pr


def kernels(backend_name):
    try:
        return load_backend(backend_name)
    except Exception:
        pytest.skip(f"{backend_name} backend not available in this environment")


def em_for(backend_name, data, theta, eta, chunk_size=None):
    norm = {
        "user": np.repeat(np.maximum(np.bincount(data[:, 0], minlength=len(theta)), 1)[:, None], theta.shape[1], axis=1),
        "item": np.repeat(np.maximum(np.bincount(data[:, 1], minlength=len(eta)), 1)[:, None], eta.shape[1], axis=1),
    }
    dims = {"n_samples": len(data), "n_user_groups": theta.shape[1], "n_item_groups": eta.shape[1], "n_ratings": 5}
    try:
        return ExpectationMaximization(dims, norm_factors=norm, backend=backend_name, chunk_size=chunk_size)
    except Exception:
        pytest.skip(f"{backend_name} backend not available in this environment")


# ---------------------------------------------------------------------------
# Kernels against the numpy reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", ["numba", "cupy"])
def test_update_coefficients_matches_numpy(backend_name):
    compute_omegas, update, prod_dist, _ = kernels(backend_name)
    omegas_np, update_np, prod_np, _ = load_backend("numpy")
    data, theta, eta, pr = random_problem(0)

    for mine, reference in zip(update(data, theta, eta, pr), update_np(data, theta, eta, pr)):
        np.testing.assert_allclose(mine, reference, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(compute_omegas(data, theta, eta, pr), omegas_np(data, theta, eta, pr), rtol=1e-12)
    np.testing.assert_allclose(prod_dist(data, theta, eta, pr), prod_np(data, theta, eta, pr), rtol=1e-9, atol=1e-12)


def test_numba_update_is_deterministic():
    _, update, _, _ = kernels("numba")
    data, theta, eta, pr = random_problem(1)
    first = update(data, theta, eta, pr)
    second = update(data, theta, eta, pr)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_numba_update_with_fewer_rows_than_threads():
    _, update, _, _ = kernels("numba")
    _, update_np, _, _ = load_backend("numpy")
    data, theta, eta, pr = random_problem(2, n=3)
    for mine, reference in zip(update(data, theta, eta, pr), update_np(data, theta, eta, pr)):
        np.testing.assert_allclose(mine, reference, rtol=1e-12, atol=1e-15)


def test_cupy_kernels_keep_arrays_on_the_gpu():
    compute_omegas, update, prod_dist, _ = kernels("cupy")
    import cupy as cp

    _, update_np, prod_np, _ = load_backend("numpy")
    data, theta, eta, pr = random_problem(3)
    device = [cp.asarray(a) for a in (data, theta, eta, pr)]

    outputs = update(*device)
    assert all(isinstance(a, cp.ndarray) for a in outputs)
    for mine, reference in zip(outputs, update_np(data, theta, eta, pr)):
        np.testing.assert_allclose(cp.asnumpy(mine), reference, rtol=1e-9, atol=1e-12)

    dist = prod_dist(*device)
    assert isinstance(dist, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(dist), prod_np(data, theta, eta, pr), rtol=1e-9)
    assert isinstance(compute_omegas(*device), cp.ndarray)

    # NumPy in, NumPy out
    assert all(isinstance(a, np.ndarray) for a in update(data, theta, eta, pr))


# ---------------------------------------------------------------------------
# Blocked E-step
# ---------------------------------------------------------------------------


def test_chunk_rows_default_keeps_the_tensor_under_budget():
    data, theta, eta, _ = random_problem(0, n=10)
    em = em_for("numpy", data, theta, eta)
    rows = em.chunk_rows(4, 4)
    assert rows * 8 * 16 <= CHUNK_BUDGET < (rows + 1) * 8 * 16
    assert em_for("numpy", data, theta, eta, chunk_size=5).chunk_rows(4, 4) == 5


@pytest.mark.parametrize("backend_name", ["numpy", "numba", "cupy"])
def test_blocked_computations_match_single_block(backend_name):
    data, theta, eta, pr = random_problem(4)
    blocked = em_for(backend_name, data, theta, eta, chunk_size=777)
    whole = em_for(backend_name, data, theta, eta, chunk_size=10**9)

    for a, b in zip(blocked.update_coefficients(data, theta, eta, pr), whole.update_coefficients(data, theta, eta, pr)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
    assert blocked.compute_likelihood(data, theta, eta, pr) == pytest.approx(whole.compute_likelihood(data, theta, eta, pr))
    assert blocked.poisson_log_likelihood(data, theta, eta, pr) == pytest.approx(
        whole.poisson_log_likelihood(data, theta, eta, pr))
    np.testing.assert_allclose(blocked.compute_prod_dist(data, theta, eta, pr), whole.compute_prod_dist(data, theta, eta, pr), rtol=1e-12)


# ---------------------------------------------------------------------------
# Whole runs
# ---------------------------------------------------------------------------


def mock_data(seed, n=2000, n_users=40, n_items=50):
    rng = np.random.default_rng(seed)
    import pandas as pd
    return pd.DataFrame(
        {
            "users": [f"user{a}" for a in rng.integers(n_users, size=n)],
            "items": [f"item{a}" for a in rng.integers(n_items, size=n)],
            "ratings": rng.integers(1, 6, size=n),
        }
    )


@pytest.mark.parametrize("backend_name", ["numba", "cupy"])
@pytest.mark.parametrize("degree_corrected", [False, True])
def test_run_matches_numpy_run(backend_name, degree_corrected):
    kernels(backend_name)
    train = DataHandler().format_train_data(mock_data(0))
    runs = {}
    for backend in ("numpy", backend_name):
        model = MMSBM(3, 2, iterations=25, seed=0, backend=backend, degree_corrected=degree_corrected)
        model._show_progress = False
        model._prepare_objects(train)
        runs[backend] = model.run_one_sampling(train, seed=11, i=0)
    for key in ("theta", "eta", "pr", "likelihood"):
        assert isinstance(runs[backend_name][key], (np.ndarray, np.floating))
        np.testing.assert_allclose(runs[backend_name][key], runs["numpy"][key], rtol=1e-6, atol=1e-10)


def test_run_in_blocks_matches_single_block():
    train = DataHandler().format_train_data(mock_data(1))
    runs = {}
    for chunk_size in (None, 300):
        model = MMSBM(3, 2, iterations=25, seed=0, backend="numpy", chunk_size=chunk_size)
        model._show_progress = False
        model._prepare_objects(train)
        runs[chunk_size] = model.run_one_sampling(train, seed=11, i=0)
    for key in ("theta", "eta", "pr", "likelihood"):
        np.testing.assert_allclose(runs[300][key], runs[None][key], rtol=1e-9, atol=1e-12)
