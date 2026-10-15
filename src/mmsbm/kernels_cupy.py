import numpy as np

try:
    import cupy as cp  # noqa: F401 – optional dependency
    from cupyx import scatter_add  # noqa: F401

    # Run a trivial operation to verify that the CUDA runtime is functional
    _ = cp.arange(1)

except Exception as _cupy_err:  # pragma: no cover – handled by backend loader
    # Any problem (missing package, missing CUDA driver, etc.) converts into a
    # plain ImportError so that `backend.load_backend()` can gracefully fall
    # back to a slower implementation.
    raise ImportError(
        "CuPy backend selected, but CuPy or a functional CUDA toolkit is not "
        "available."
    ) from _cupy_err

"""GPU kernels.

Every kernel accepts NumPy or CuPy arrays.  CuPy arrays are used in place and
the results stay on the GPU; NumPy arrays are copied to the GPU and the
results copied back.  Which one happens is decided by ``theta``: the EM loop
keeps the data and the parameters on the GPU for the whole run and only moves
the final parameters back, while callers that hold NumPy arrays (prediction,
tests) keep working unchanged.
"""

__all__ = [
    "compute_omegas",
    "update_coefficients",
    "prod_dist",
]

_EPS = np.finfo(np.float64).eps

# cupy.add.at replaced cupyx.scatter_add (deprecated); keep the fallback for old CuPy versions
_scatter_add = cp.add.at if hasattr(cp.add, "at") else scatter_add


def _like(reference, value):
    """``value`` on the GPU if ``reference`` is a CuPy array, as NumPy otherwise."""
    return value if isinstance(reference, cp.ndarray) else cp.asnumpy(value)


def _omegas(data, theta, eta, pr):
    """Unnormalised responsibilities, all arrays on the GPU."""
    user_idx = data[:, 0]
    item_idx = data[:, 1]
    rating_idx = data[:, 2]
    pr_T = pr.transpose(2, 0, 1)
    return (
        theta[user_idx][:, :, None] *
        eta[item_idx][:, None, :] *
        pr_T[rating_idx]
    )


def compute_omegas(data, theta, eta, pr):
    """The full (N, K, L) tensor of unnormalised responsibilities."""
    omegas = _omegas(cp.asarray(data), cp.asarray(theta), cp.asarray(eta), cp.asarray(pr))
    return _like(theta, omegas)


def update_coefficients(data, theta, eta, pr):
    """Unnormalised numerators of theta, eta and pr, accumulated on the GPU.

    The accumulations use atomic scatter-adds, so the order of the summations
    is not fixed and results can differ in the last bits between calls.
    """
    data_d = cp.asarray(data)
    theta_d = cp.asarray(theta)
    eta_d = cp.asarray(eta)
    pr_d = cp.asarray(pr)

    omegas = _omegas(data_d, theta_d, eta_d, pr_d)
    sum_omega = omegas.sum(axis=(1, 2))
    increments = omegas / cp.maximum(sum_omega, _EPS)[:, None, None]

    n_theta = cp.zeros_like(theta_d)
    _scatter_add(n_theta, data_d[:, 0], increments.sum(axis=2))

    n_eta = cp.zeros_like(eta_d)
    _scatter_add(n_eta, data_d[:, 1], increments.sum(axis=1))

    K, L, R = pr_d.shape
    n_pr_T = cp.zeros((R, K, L), dtype=pr_d.dtype)
    _scatter_add(n_pr_T, data_d[:, 2], increments)
    n_pr = cp.ascontiguousarray(n_pr_T.transpose(1, 2, 0))

    return _like(theta, n_theta), _like(theta, n_eta), _like(theta, n_pr)


def prod_dist(data, theta, eta, pr):
    """Rating distribution p(r | u, i) for all rows in *data*."""
    data_d = cp.asarray(data)
    theta_d = cp.asarray(theta)
    eta_d = cp.asarray(eta)
    pr_d = cp.asarray(pr)

    result = cp.einsum('nk,nl,klr->nr',
                       theta_d[data_d[:, 0]],
                       eta_d[data_d[:, 1]],
                       pr_d)
    return _like(theta, result)
