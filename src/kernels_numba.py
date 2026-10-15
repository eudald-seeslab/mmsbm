import numba
from numba import njit, prange
import numpy as np

"""Numba-accelerated computational kernels.

These have the same public API as `kernels_numpy.py` but are compiled with
Numba's `njit` for faster CPU execution.  The compiled code is cached on disk
(``cache=True``), so a process only pays the compilation once: worker
processes started for ``sampling > 1`` reuse the cache instead of recompiling.

``update_coefficients`` never materialises the responsibility tensor: every
row computes its K x L block, normalises it and adds it to per-thread partial
accumulators, which are summed at the end.  Its memory is therefore
independent of the number of ratings.
"""

__all__ = [
    "compute_omegas",
    "update_coefficients",
    "prod_dist",
]

_EPS = np.finfo(np.float64).eps

# Upper bound, in bytes, of the per-thread partial accumulators of
# update_coefficients (one copy of theta, eta and pr per chunk).
ACCUMULATOR_BUDGET = 256 * 1024 * 1024


@njit(parallel=True, fastmath=True, cache=True)
def compute_omegas(data, theta, eta, pr):
    """Numba version of compute_omegas: the full (N, K, L) tensor."""
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


@njit(parallel=True, fastmath=True, cache=True)
def _partial_updates(data, theta, eta, pr, n_chunks):
    """Per-chunk numerators of theta, eta and pr; chunks run in parallel."""
    n = data.shape[0]
    U, K = theta.shape
    I, L = eta.shape
    R = pr.shape[2]
    chunk = (n + n_chunks - 1) // n_chunks

    part_theta = np.zeros((n_chunks, U, K))
    part_eta = np.zeros((n_chunks, I, L))
    part_pr = np.zeros((n_chunks, K, L, R))

    for c in prange(n_chunks):
        local = np.empty((K, L))
        start = c * chunk
        end = min(n, start + chunk)
        for idx in range(start, end):
            u = data[idx, 0]
            i = data[idx, 1]
            r = data[idx, 2]

            total = 0.0
            for k in range(K):
                t_uk = theta[u, k]
                for l in range(L):
                    w = t_uk * eta[i, l] * pr[k, l, r]
                    local[k, l] = w
                    total += w
            total = max(total, _EPS)

            for k in range(K):
                for l in range(L):
                    inc = local[k, l] / total
                    part_theta[c, u, k] += inc
                    part_eta[c, i, l] += inc
                    part_pr[c, k, l, r] += inc

    return part_theta, part_eta, part_pr


def update_coefficients(data, theta, eta, pr):
    """Parallel Numba version of update_coefficients.

    The rows are split into as many contiguous chunks as there are Numba
    threads (fewer if the per-chunk accumulators would exceed
    ``ACCUMULATOR_BUDGET``), each chunk accumulates into its own copy of the
    numerators and the copies are summed.  The result is deterministic for a
    given number of threads; it differs from the serial order of summation in
    the last bits.
    """
    n = data.shape[0]
    U, K = theta.shape
    I, L = eta.shape
    R = pr.shape[2]

    per_chunk = 8 * (U * K + I * L + K * L * R)
    n_chunks = max(1, min(numba.get_num_threads(), ACCUMULATOR_BUDGET // per_chunk, n))

    part_theta, part_eta, part_pr = _partial_updates(data, theta, eta, pr, n_chunks)
    return part_theta.sum(axis=0), part_eta.sum(axis=0), part_pr.sum(axis=0)


@njit(parallel=True, fastmath=True, cache=True)
def prod_dist(data, theta, eta, pr):
    """Rating distribution p(r | u, i) for all rows in *data*."""
    n = data.shape[0]
    K, L, R = pr.shape

    out = np.zeros((n, R))

    for idx in prange(n):
        u = data[idx, 0]
        i = data[idx, 1]
        for k in range(K):
            t_uk = theta[u, k]
            for l in range(L):
                w = t_uk * eta[i, l]
                for r in range(R):
                    out[idx, r] += w * pr[k, l, r]

    return out
