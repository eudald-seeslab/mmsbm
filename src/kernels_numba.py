from numba import njit, prange
import numpy as np

"""Numba-accelerated computational kernels.

These have the same public API as `kernels_numpy.py` but are compiled with
Numba's `njit` for faster CPU execution.  Broadcasting-heavy expressions are
kept because Numba now supports most NumPy operations; when support is
missing we fall back to simple loops.
"""

__all__ = [
    "compute_omegas",
    "update_coefficients",
    "prod_dist",
]


@njit(parallel=True, fastmath=True)
def compute_omegas(data, theta, eta, pr):
    """Numba version of compute_omegas."""
    n = data.shape[0]
    K = theta.shape[1]
    L = eta.shape[1]

    omegas = np.empty((n, K, L))

    for idx in prange(n):
        u = data[idx, 0]
        i = data[idx, 1]
        r = data[idx, 2]
        for k in range(K):
            t_uk = theta[u, k]
            for l in range(L):
                omegas[idx, k, l] = t_uk * eta[i, l] * pr[k, l, r]

    return omegas


@njit
def _sum_axis_1_2(arr):
    """Helper to compute arr.sum(axis=(1, 2)) for older Numba versions."""
    n_rows = arr.shape[0]
    out = np.empty(n_rows, dtype=arr.dtype)
    for i in range(n_rows):
        out[i] = arr[i, :, :].sum()
    return out


@njit(fastmath=True)
def update_coefficients(data, theta, eta, pr):
    """Serial Numba version of update_coefficients."""
    omegas = compute_omegas(data, theta, eta, pr)
    sum_omega = _sum_axis_1_2(omegas)

    # Avoid division by zero
    eps = np.finfo(np.float64).eps
    increments = omegas / (sum_omega + eps)[:, None, None]

    n_theta = np.zeros_like(theta)
    n_eta = np.zeros_like(eta)
    n_pr = np.zeros_like(pr)

    K, L, R = pr.shape

    # This loop is serial; prange is not safe here without atomic operations
    for idx in range(data.shape[0]):
        u, i, r = data[idx, 0], data[idx, 1], data[idx, 2]

        # Update theta for user u (sum over l)
        for k in range(K):
            n_theta[u, k] += np.sum(increments[idx, k, :])

        # Update eta for item i (sum over k)
        for l in range(L):
            n_eta[i, l] += np.sum(increments[idx, :, l])

        # Update pr for rating r (no sum needed, just add increments)
        for k in range(K):
            for l in range(L):
                # .item() is crucial to convert 0-d array to scalar for Numba
                n_pr[k, l, r] += increments[idx, k, l].item()

    return n_theta, n_eta, n_pr


from kernels_numpy import prod_dist  # Numba lacks einsum; use NumPy version 