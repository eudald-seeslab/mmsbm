import logging
from datetime import datetime
from os import path
import numpy as np
from tqdm import tqdm

from lib.funcs import (
    compute_likelihood,
    compute_prod_dist,
    compute_indicators,
    compute_final_stats,
    update_coefs,
    normalize_with_d,
    normalize_with_self,
    init_random_array
)
from lib.utils import parse_args

# Suppress numba warnings (for now)
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def main():

    start_time = datetime.now()
    train_set, test_set, user_groups, item_groups, iterations, sampling, seed = parse_args()

    # Initiate the random state
    rng = np.random.default_rng(seed)

    logger = logging.getLogger("MMSBM")
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Running {sampling} runs of {iterations} iterations.")

    # Get data
    train = np.genfromtxt(path.join("data", train_set), delimiter="\t", usecols=[0, 1, 2], dtype="int")
    test = np.genfromtxt(path.join("data", test_set), delimiter="\t", usecols=[0, 1, 2], dtype="int")

    # Create a few dicts with the relationships
    # TODO: think whether initialization with 0 is needed
    d0 = {0: []}
    d1 = {0: []}
    [d0.update({a: list(train[train[:, 0] == a, 1])}) for a in set(train[:, 0])]
    [d1.update({a: list(train[train[:, 1] == a, 0])}) for a in set(train[:, 1])]
    ratings = sorted(set(train[:, 2]))
    r = len(ratings)
    rat = np.zeros((test.shape[0], r))
    p = int(train[:, 0].max())
    m = int(train[:, 1].max())

    # If, for some reason, there are missing links, we need to fill them:
    [d0.update({a: []}) for a in set(range(p + 1)).difference(set(d0.keys()))]
    [d1.update({a: []}) for a in set(range(m + 1)).difference(set(d1.keys()))]

    likelihoods = []
    for s in range(sampling):

        logger.info(f"Running run {s + 1} out of {sampling}.")

        # Generate random (but normalized) inits
        theta = normalize_with_d(
            init_random_array((p + 1, user_groups), rng), d0
        )
        eta = normalize_with_d(
            init_random_array((m + 1, item_groups), rng), d1
        )
        pr = normalize_with_self(
            init_random_array((user_groups, item_groups, r), rng)
        )

        # Do the work
        for _ in tqdm(range(iterations)):
            # This is the crux of the script; please see funcs.py
            n_theta, n_eta, npr = update_coefs(data=train, ratings=ratings, theta=theta, eta=eta, pr=pr)

            # Update with normalization
            theta = normalize_with_d(n_theta, d0)
            eta = normalize_with_d(n_eta, d1)
            pr = normalize_with_self(npr)

        # Compute the likelihood
        likelihoods.append(compute_likelihood(train, ratings, theta, eta, pr))

        # Predictions
        rat += compute_prod_dist(test, theta, eta, pr, sampling)

    # How did we do?
    rat = compute_indicators(rat, test, ratings)
    # Final model quality indicators
    accuracy, mae, s2, s2pond = compute_final_stats(rat)

    final_time = datetime.now()
    logger.info(f"Done {sampling} runs in {(final_time - start_time).total_seconds() / 60.0:.2f} minutes.")
    logger.info(f"We had an accuracy of {accuracy}, a MAE of {mae} and s2 and weighted s2 of {s2} and {s2pond:.0f}.")


if __name__ == main():
    main()
