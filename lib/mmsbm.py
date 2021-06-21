import logging
import multiprocessing
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from lib.funcs import (
    compute_indicators,
    compute_final_stats,
    normalize_with_d,
    init_random_array,
    normalize_with_self,
    update_coefs,
    compute_likelihood,
    compute_prod_dist,
)
from lib.utils import get_data, check_data


def mmsbm(
    train_set,
    test_set,
    user_groups,
    item_groups,
    iterations,
    sampling,
    seed,
    notebook=False,
):
    start_time = datetime.now()

    # Initiate the random state
    rng = np.random.default_rng(seed)
    # Create seeds for each process
    seeds = list(rng.integers(low=1, high=10000, size=sampling))

    logger = logging.getLogger("MMSBM")
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Running {sampling} runs of {iterations} iterations.")

    # Get data
    data_dir = os.path.join(os.getcwd(), "data")
    train = get_data(os.path.join(data_dir, train_set))
    check_data(train)
    test = get_data(os.path.join(data_dir, test_set))
    check_data(train)

    # Create a few dicts with the relationships
    # TODO: think whether initialization with 0 is needed
    d0 = {0: []}
    d1 = {0: []}
    [d0.update({a: list(train[train[:, 0] == a, 1])}) for a in set(train[:, 0])]
    [d1.update({a: list(train[train[:, 1] == a, 0])}) for a in set(train[:, 1])]
    ratings = sorted(set(train[:, 2]))
    r = len(ratings)
    p = int(train[:, 0].max())
    m = int(train[:, 1].max())

    # If, for some reason, there are missing links, we need to fill them:
    [d0.update({a: []}) for a in set(range(p + 1)).difference(set(d0.keys()))]
    [d1.update({a: []}) for a in set(range(m + 1)).difference(set(d1.keys()))]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(sampling):
        proc = multiprocessing.Process(
            target=run_one_sampling,
            args=(
                d0,
                d1,
                p,
                m,
                r,
                user_groups,
                item_groups,
                iterations,
                train,
                test,
                ratings,
                seeds[i],
                i,
                return_dict,
            ),
        )
        jobs.append(proc)
        proc.start()

    for proc in jobs:
        proc.join()

    rat = np.array([a["rat"] for a in return_dict.values()]).mean(axis=0)
    prs = [a["prs"] for a in return_dict.values()]

    # How did we do?
    rat = compute_indicators(rat, test, ratings)
    # Final model quality indicators
    accuracy, mae, s2, s2pond = compute_final_stats(rat)

    final_time = datetime.now()
    logger.info(
        f"Done {sampling} runs in {(final_time - start_time).total_seconds() / 60.0:.2f} minutes."
    )
    logger.info(
        f"We had an accuracy of {accuracy}, a MAE of {mae} and s2 and weighted s2 of {s2} and {s2pond:.0f}."
    )

    # In case we are running from a notebook, and we want to inspect the results
    if notebook:
        return prs, accuracy, mae, s2, s2pond, rat
    else:
        return accuracy


def run_one_sampling(
    d0,
    d1,
    p,
    m,
    r,
    user_groups,
    item_groups,
    iterations,
    train,
    test,
    ratings,
    seed,
    i,
    return_dict,
):
    rng = np.random.default_rng(seed)

    # Generate random (but normalized) inits
    theta = normalize_with_d(init_random_array((p + 1, user_groups), rng), d0)
    eta = normalize_with_d(init_random_array((m + 1, item_groups), rng), d1)
    pr = normalize_with_self(init_random_array((user_groups, item_groups, r), rng))

    # Do the work
    # We store the prs to check convergence
    prs = []
    for _ in tqdm(range(iterations)):
        # This is the crux of the script; please see funcs.py
        n_theta, n_eta, npr = update_coefs(
            data=train, ratings=ratings, theta=theta, eta=eta, pr=pr
        )

        # Update with normalization
        theta = normalize_with_d(n_theta, d0)
        eta = normalize_with_d(n_eta, d1)
        pr = normalize_with_self(npr)

        # This can be removed when not debugging
        prs.append(pr)

    likelihood = compute_likelihood(train, ratings, theta, eta, pr)
    rat = compute_prod_dist(test, theta, eta, pr)

    return_dict[i] = {"likelihood": likelihood, "rat": rat, "prs": prs}

    return None
