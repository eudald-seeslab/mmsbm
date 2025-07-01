# benchmark_mmsbm.py
import argparse
import time
from typing import Tuple

import numpy as np
import pandas as pd

from mmsbm import MMSBM


# ------------------------------ helpers ---------------------------------

def mock_data(seed: int = 0,
              n_obs: int = 20_000,
              n_users: int = 1_000,
              n_items: int = 1_500,
              rating_scale: Tuple[int, int] = (1, 5)) -> pd.DataFrame:
    """Generate a random implicit–feedback dataset like the unit-tests.

    Returned columns: ``users``, ``items``, ``ratings`` (strings for first two).
    """
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "users": rng.integers(0, n_users, size=n_obs).astype(str),
            "items": rng.integers(0, n_items, size=n_obs).astype(str),
            "ratings": rng.integers(rating_scale[0], rating_scale[1] + 1, size=n_obs),
        }
    )


# ------------------------------ benchmark --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark MMSBM training and prediction speed.")
    parser.add_argument("--n_obs", type=int, default=20_000, help="Number of observations to simulate.")
    parser.add_argument("--iterations", type=int, default=50, help="EM iterations per run.")
    parser.add_argument("--user_groups", type=int, default=2, help="Number of user latent groups.")
    parser.add_argument("--item_groups", type=int, default=2, help="Number of item latent groups.")
    parser.add_argument("--sampling", type=int, default=1, help="Parallel sampling runs.")
    parser.add_argument("--backend", type=str, default="numpy", help="Backend to use. Options: numpy, cupy, numba.")
    args = parser.parse_args()

    # Generate data
    data = mock_data(n_obs=args.n_obs)

    model = MMSBM(
        user_groups=args.user_groups,
        item_groups=args.item_groups,
        iterations=args.iterations,
        sampling=args.sampling,
        seed=0,
        backend=args.backend,
        debug=True
    )

    # --- training time ---------------------------------------------------
    t0 = time.perf_counter()
    model.fit(data, silent=True)
    train_time = time.perf_counter() - t0

    # --- prediction time -------------------------------------------------
    t1 = time.perf_counter()
    _ = model.predict(data)  # predict on same data; size identical across runs
    pred_time = time.perf_counter() - t1

    print(
        f"Training time:   {train_time:8.3f} s\n"
        f"Prediction time: {pred_time:8.3f} s\n"
        f"Total time:      {train_time + pred_time:8.3f} s"
    )


if __name__ == "__main__":
    main()
