import numpy as np
import pytest

from backend import load_backend


@pytest.mark.parametrize("backend_name", ["numba", "cupy"])
def test_backend_compute_omegas_matches_numpy(backend_name):
    """Ensure alternative backends return the same omega tensor as NumPy implementation.

    The test is skipped automatically if the requested backend cannot be loaded
    (e.g. Numba or CuPy is not installed / CUDA not available).
    """
    try:
        compute_backend, _, _, loaded_name = load_backend(backend_name)
    except Exception:  # pragma: no cover – fallback for environments without the backend
        pytest.skip(f"{backend_name} backend not available in this environment")

    compute_numpy, _, _, _ = load_backend("numpy")

    rng = np.random.default_rng(0)
    data = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 2]], dtype=np.int64)
    theta = rng.random((2, 2))
    eta = rng.random((2, 2))
    pr = rng.random((2, 2, 3))

    # Make sure distributions are normalised for realistic conditions
    theta /= theta.sum(axis=1, keepdims=True)
    eta /= eta.sum(axis=1, keepdims=True)
    pr /= pr.sum(axis=2, keepdims=True)

    out_numpy = compute_numpy(data, theta, eta, pr)
    out_backend = compute_backend(data, theta, eta, pr)

    assert np.allclose(out_numpy, out_backend, atol=1e-8)


@pytest.mark.parametrize("backend_name", ["numba", "cupy"])
def test_backend_prod_dist_matches_numpy(backend_name):
    """Ensure prod_dist contract matches NumPy reference."""
    try:
        _, _, prod_backend, _ = load_backend(backend_name)
    except Exception:
        pytest.skip(f"{backend_name} backend not available in this environment")

    _, _, prod_numpy, _ = load_backend("numpy")

    rng = np.random.default_rng(1)
    data = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 2]], dtype=np.int64)
    theta = rng.random((2, 2))
    eta = rng.random((2, 2))
    pr = rng.random((2, 2, 3))

    theta /= theta.sum(axis=1, keepdims=True)
    eta /= eta.sum(axis=1, keepdims=True)
    pr /= pr.sum(axis=2, keepdims=True)

    dist_np = prod_numpy(data, theta, eta, pr)
    dist_back = prod_backend(data, theta, eta, pr)

    assert np.allclose(dist_np, dist_back, atol=1e-8) 