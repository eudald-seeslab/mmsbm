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

__all__ = [
    "compute_omegas",
    "update_coefficients",
    "prod_dist",
]


def compute_omegas(data, theta, eta, pr):
    """
    Public-facing compute_omegas for CuPy.
    Accepts numpy arrays and returns a numpy array.
    """
    # Move data to GPU
    data_cp = cp.asarray(data)
    theta_cp = cp.asarray(theta)
    eta_cp = cp.asarray(eta)
    pr_cp = cp.asarray(pr)

    # Perform computation on GPU
    user_idx = data_cp[:, 0]
    item_idx = data_cp[:, 1]
    rating_idx = data_cp[:, 2]
    pr_T = pr_cp.transpose(2, 0, 1)
    omegas_cp = (
        theta_cp[user_idx][:, :, None] *
        eta_cp[item_idx][:, None, :] *
        pr_T[rating_idx]
    )

    # Move result back to CPU
    return cp.asnumpy(omegas_cp)


def update_coefficients(data, theta, eta, pr):
    """GPU-accelerated version of update_coefficients."""
    # Move data to GPU
    data_cp = cp.asarray(data)
    theta_cp = cp.asarray(theta)
    eta_cp = cp.asarray(eta)
    pr_cp = cp.asarray(pr)

    # --- Internal omega calculation (all on GPU) ---
    user_idx, item_idx, rating_idx = data_cp[:, 0], data_cp[:, 1], data_cp[:, 2]
    pr_T = pr_cp.transpose(2, 0, 1)
    omegas = (
        theta_cp[user_idx][:, :, None] *
        eta_cp[item_idx][:, None, :] *
        pr_T[rating_idx]
    )
    sum_omega = omegas.sum(axis=(1, 2))
    eps = cp.finfo(cp.float64).eps
    increments = omegas / (sum_omega + eps)[:, None, None]

    # --- Parameter updates (all on GPU) ---
    n_theta = cp.zeros_like(theta_cp)
    n_eta = cp.zeros_like(eta_cp)
    n_pr = cp.zeros_like(pr_cp)

    # Theta update
    inc_theta = increments.sum(axis=2)
    scatter_add(n_theta, user_idx, inc_theta)

    # Eta update
    inc_eta = increments.sum(axis=1)
    scatter_add(n_eta, item_idx, inc_eta)

    # Pr update
    K, L, R = n_pr.shape
    for r in range(R):
        mask = (rating_idx == r)
        if mask.any():
            n_pr[:, :, r] = increments[mask].sum(axis=0)

    # Move results back to CPU
    return cp.asnumpy(n_theta), cp.asnumpy(n_eta), cp.asnumpy(n_pr)


def prod_dist(data, theta, eta, pr):
    """GPU-accelerated version of prod_dist."""
    # Move data to GPU
    data_cp = cp.asarray(data)
    theta_cp = cp.asarray(theta)
    eta_cp = cp.asarray(eta)
    pr_cp = cp.asarray(pr)

    user_idx = data_cp[:, 0]
    item_idx = data_cp[:, 1]

    # Perform computation on GPU
    result = cp.einsum('nk,nl,klr->nr',
                       theta_cp[user_idx],
                       eta_cp[item_idx],
                       pr_cp)

    # Move result back to CPU
    return cp.asnumpy(result) 