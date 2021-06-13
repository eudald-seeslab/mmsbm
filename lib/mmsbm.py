import logging
import multiprocessing
import os
from datetime import datetime

import numpy as np

from lib.funcs import compute_indicators, compute_final_stats

from lib.one_sampling import run_one_sampling


def mmsbm(train_set, test_set, user_groups, item_groups, iterations, sampling, seed):
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
    train = np.genfromtxt(os.path.join(data_dir, train_set), delimiter="\t", usecols=[0, 1, 2], dtype="int")
    test = np.genfromtxt(os.path.join(data_dir, test_set), delimiter="\t", usecols=[0, 1, 2], dtype="int")

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

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(sampling):
        pr = multiprocessing.Process(
                target=run_one_sampling,
                args=(d0, d1, p, m, r, user_groups, item_groups, iterations, train, test, ratings, seeds[i], i, return_dict)
            )
        jobs.append(pr)
        pr.start()

    for proc in jobs:
        proc.join()

    rat = np.array([a["rat"] for a in return_dict.values()]).mean(axis=0)
    prs = [a["prs"] for a in return_dict.values()]

    # How did we do?
    rat = compute_indicators(rat, test, ratings)
    # Final model quality indicators
    accuracy, mae, s2, s2pond = compute_final_stats(rat)

    final_time = datetime.now()
    logger.info(f"Done {sampling} runs in {(final_time - start_time).total_seconds() / 60.0:.2f} minutes.")
    logger.info(f"We had an accuracy of {accuracy}, a MAE of {mae} and s2 and weighted s2 of {s2} and {s2pond:.0f}.")

    # In case we are running from a notebook and we want to inspect the results
    return prs, accuracy, mae, s2, s2pond