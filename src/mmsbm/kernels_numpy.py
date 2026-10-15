import numpy as np

"""Numpy reference implementations of the three computational kernels used by
ExpectationMaximization.  Each function operates purely on NumPy ndarrays and
is designed to have the exact same call-signature across all back-ends
(NumPy, Numba, CuPy, …) so that the high-level EM driver can dispatch to it
without if/else logic.
"""

__all__ = [
    "compute_omegas",
    "update_coefficients",
    "prod_dist",
]


# ---------------------------------------------------------------------------
# ω responsibilities
# ---------------------------------------------------------------------------

def compute_omegas(data: np.ndarray,
                   theta: np.ndarray,
                   eta: np.ndarray,
                   pr: np.ndarray) -> np.ndarray:
    """Compute unnormalised responsibilities ω_{ui}(k,ℓ) in pure NumPy."""
    user_idx = data[:, 0]
    item_idx = data[:, 1]
    rating_idx = data[:, 2]

    pr_T = pr.transpose(2, 0, 1)  # (R, K, L)

    return (
        theta[user_idx][:, :, None] *
        eta[item_idx][:, None, :] *
        pr_T[rating_idx]
    )


# ---------------------------------------------------------------------------
# M-step parameter updates (unnormalised numerators)
# ---------------------------------------------------------------------------

def update_coefficients(data: np.ndarray,
                        theta: np.ndarray,
                        eta: np.ndarray,
                        pr: np.ndarray):
    """Return unnormalised updates (Σω) for θ, η and p in pure NumPy."""
    omegas = compute_omegas(data, theta, eta, pr)
    sum_omega = omegas.sum(axis=(1, 2))

    eps = np.finfo(float).eps
    increments = omegas / np.maximum(sum_omega, eps)[:, None, None]

    user_idx = data[:, 0]
    item_idx = data[:, 1]
    rating_idx = data[:, 2]

    K = theta.shape[1]
    L = eta.shape[1]
    R = pr.shape[2]

    # θ update ---------------------------------------------------------------
    inc_theta = increments.sum(axis=2)              # (N, K)
    n_theta = np.zeros_like(theta)
    np.add.at(n_theta, user_idx, inc_theta)

    # η update ---------------------------------------------------------------
    inc_eta = increments.sum(axis=1)                # (N, L)
    n_eta = np.zeros_like(eta)
    np.add.at(n_eta, item_idx, inc_eta)

    # p update ---------------------------------------------------------------
    n_pr = np.zeros_like(pr)
    for r in range(R):
        mask = rating_idx == r
        if np.any(mask):
            n_pr[:, :, r] = increments[mask].sum(axis=0)

    return n_theta, n_eta, n_pr


# ---------------------------------------------------------------------------
# Product distribution for prediction
# ---------------------------------------------------------------------------

def prod_dist(data: np.ndarray,
              theta: np.ndarray,
              eta: np.ndarray,
              pr: np.ndarray) -> np.ndarray:
    """Vectorised rating distribution p(r | u,i) for all rows in *data*."""
    user_idx = data[:, 0]
    item_idx = data[:, 1]

    return np.einsum('nk,nl,klr->nr',
                     theta[user_idx],
                     eta[item_idx],
                     pr) 