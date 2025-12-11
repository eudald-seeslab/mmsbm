import sys

# Numba backend unsupported on Windows (native llvmlite/Numba crashes).
if sys.platform.startswith(("win", "cygwin")):
    raise ImportError("Numba backend is not supported on Windows.")

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


@njit(fastmath=True)
def update_coefficients(data, theta, eta, pr):
    """Serial Numba version of update_coefficients."""
    omegas = compute_omegas(data, theta, eta, pr)

    n_samples = data.shape[0]
    K = theta.shape[1]
    L = eta.shape[1]
    R = pr.shape[2]

    # Sum ω over (k,l) using NumPy reduction inside njit
    sum_omega = omegas.sum(axis=(1, 2))

    # Avoid division by zero
    eps = np.finfo(np.float64).eps
    increments = omegas / (sum_omega + eps)[:, None, None]

    n_theta = np.zeros_like(theta)
    n_eta = np.zeros_like(eta)
    n_pr = np.zeros_like(pr)

    for idx in range(n_samples):
        u = data[idx, 0]
        i = data[idx, 1]
        r = data[idx, 2]

        inc = increments[idx]
        # θ: sum over l
        n_theta[u] += inc.sum(axis=1)
        # η: sum over k
        n_eta[i] += inc.sum(axis=0)
        # p: accumulate full K×L slice for rating r
        n_pr[:, :, r] += inc

    return n_theta, n_eta, n_pr


from kernels_numpy import prod_dist  # Numba lacks einsum; use NumPy version 