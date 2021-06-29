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


class MMSBM:
    def __init__(
        self,
        train_set,
        test_set,
        user_groups,
        item_groups,
        iterations,
        sampling,
        seed,
        notebook=False,
    ):
        self.start_time = datetime.now()

        # Initiate the random state
        rng = np.random.default_rng(seed)
        # Create seeds for each process
        self.seeds = list(rng.integers(low=1, high=10000, size=sampling))

        self.logger = logging.getLogger("MMSBM")
        logging.basicConfig(level=logging.DEBUG if notebook else logging.INFO)
        self.logger.info(f"Running {sampling} runs of {iterations} iterations.")

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
        self.ratings = sorted(set(train[:, 2]))
        self.r = len(self.ratings)
        self.p = int(train[:, 0].max())
        self.m = int(train[:, 1].max())

        # If, for some reason, there are missing links, we need to fill them:
        [d0.update({a: []}) for a in set(range(self.p + 1)).difference(set(d0.keys()))]
        [d1.update({a: []}) for a in set(range(self.m + 1)).difference(set(d1.keys()))]

        self.train = train
        self.test = test
        self.d0 = d0
        self.d1 = d1
        self.sampling = sampling
        self.user_groups = user_groups
        self.item_groups = item_groups
        self.iterations = iterations
        self.notebook = notebook

    def process(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(self.sampling):
            proc = multiprocessing.Process(
                target=self.run_one_sampling,
                args=(self.seeds[i], i, return_dict),
            )
            jobs.append(proc)
            proc.start()

        for proc in jobs:
            proc.join()

        return return_dict

    def postprocess(self, return_dict):

        rat = np.array([a["rat"] for a in return_dict.values()]).mean(axis=0)
        prs = [a["prs"] for a in return_dict.values()]
        lkh = [a["likelihood"] for a in return_dict.values()]
        theta = [a["theta"] for a in return_dict.values()]
        eta = [a["eta"] for a in return_dict.values()]

        # How did we do?
        rat = compute_indicators(rat, self.test, self.ratings)
        # Final model quality indicators
        accuracy, mae, s2, s2pond = compute_final_stats(rat)

        final_time = datetime.now()
        self.logger.info(
            f"Done {self.sampling} runs in {(final_time - self.start_time).total_seconds() / 60.0:.2f} minutes."
        )
        self.logger.info(
            f"We had an accuracy of {accuracy}, a MAE of {mae} and s2 and weighted s2 of {s2} and {s2pond:.0f}."
        )

        # In case we are running from a notebook, and we want to inspect the results
        if self.notebook:
            return prs, accuracy, mae, s2, s2pond, rat, lkh, theta, eta
        else:
            return accuracy

    def run_one_sampling(self, seed, i, return_dict):
        rng = np.random.default_rng(seed)

        # Generate random (but normalized) inits
        theta = normalize_with_d(
            init_random_array((self.p + 1, self.user_groups), rng), self.d0
        )
        eta = normalize_with_d(
            init_random_array((self.m + 1, self.item_groups), rng), self.d1
        )
        pr = normalize_with_self(
            init_random_array((self.user_groups, self.item_groups, self.r), rng)
        )

        # Do the work
        # We store the prs to check convergence
        prs = []
        for i in tqdm(range(self.iterations)):
            # This is the crux of the script; please see funcs.py
            n_theta, n_eta, npr = update_coefs(
                data=self.train, ratings=self.ratings, theta=theta, eta=eta, pr=pr
            )

            # Update with normalization
            theta = normalize_with_d(n_theta, self.d0)
            eta = normalize_with_d(n_eta, self.d1)
            pr = normalize_with_self(npr)

            # This can be removed when not debugging
            prs.append(pr)

            """
            # For debugging purposes; compute likelihood every once in a while
            if i % 50 == 0:
                likelihood = compute_likelihood(self.train, self.ratings, theta, eta, pr)
                # FIXME: convert to logger
                print(f"\nLikelihood at run {i} is {likelihood.sum():.0f}")
            """

        likelihood = compute_likelihood(self.train, self.ratings, theta, eta, pr)
        rat = compute_prod_dist(self.test, theta, eta, pr)

        return_dict[i] = {"likelihood": likelihood, "rat": rat, "prs": prs, "theta": theta, "eta": eta}

        return None
