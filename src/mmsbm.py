import logging
import multiprocessing
from datetime import datetime
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data_handler import DataHandler
from expectation_maximization import (
    normalize_with_d,
    update_coefficients,
    normalize_with_self,
    compute_likelihood,
    compute_prod_dist,
)
from helpers import structure_folds, get_n_per_group


class MMSBM:
    """
    Mixed Membership Stochastic Block Model.

    This model computes the parameters of a MMSBM via expectation-maximization (EM) algorithm.

    Parameters
    ----------
    user_groups : int
        The number of groups in which to classify the users.

    item_groups: int
        The number of groups in which to classify the items.

    iterations: int, default=400
        The number of iterations to run for each EM run.

    sampling: int, default=1
        The number of parallel computations to run. Each computation leads to slightly different results so a sampling
        bigger than 0 adds to robustness of the model.

    seed: int, default=1714
        Seed for reproducibility.

    debug: int, default=False
        Make everything more verbose and set iterations to 10 and sampling to 1.

    Attributes
    ---------
    results: dictionary
        Contains the goodness of fit statistics and the computed objects. Please see the score function for more
        details.

    """

    data_handler = None
    results = None
    test = None
    theta = None
    eta = None
    pr = None
    likelihood = None
    prediction_matrix = None

    def __init__(
        self,
        user_groups,
        item_groups,
        iterations=400,
        sampling=1,
        seed=None,
        debug=False,
    ):
        self.start_time = datetime.now()
        self.user_groups = user_groups
        self.item_groups = item_groups
        self.iterations = iterations
        self.sampling = sampling
        self.debug = debug

        # Initiate the general random state
        rng = np.random.default_rng(seed)
        # Initiate the child random states
        ss = rng.bit_generator._seed_seq
        self.child_states = ss.spawn(sampling)

        self.logger = logging.getLogger("MMSBM")
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    def _prepare_objects(self, train):

        # Create a few dicts with the relationships
        d0 = {}
        d1 = {}
        [d0.update({a: list(train[train[:, 0] == a, 1])}) for a in set(train[:, 0])]
        [d1.update({a: list(train[train[:, 1] == a, 0])}) for a in set(train[:, 1])]
        self.ratings = sorted(set(train[:, 2]))
        self.r = max(self.ratings)
        self.p = int(train[:, 0].max())
        self.m = int(train[:, 1].max())

        # If, for some reason, there are missing links, we need to fill them:
        # TODO: I think this can be safely removed
        [d0.update({a: []}) for a in set(range(self.p)).difference(set(d0.keys()))]
        [d1.update({a: []}) for a in set(range(self.m)).difference(set(d1.keys()))]

        self.train = train
        self.d0 = d0
        self.d1 = d1

    def fit(self, data, silent=False):
        """
        Fit the MMSBM with the given data.
        :param data: {dataframe} of shape (n_samples, 3)
            The input data. The first column has to be the user identifier, the second column the item identifier and
            the third the rating.
        :param silent: boolean.
            Do you want to shut off most of the notifications? (Mostly for internal use)
        :return: None
        """

        if not silent:
            self.logger.info(
                f"Running {self.sampling} runs of {self.iterations} iterations."
            )

        # Get data
        self.data_handler = DataHandler()
        train = self.data_handler.format_train_data(data)
        self._prepare_objects(train)

        with multiprocessing.Pool(processes=self.sampling) as pool:
            self.results = pool.starmap(self.run_one_sampling, zip(repeat(train), self.child_states, list(range(self.sampling))))

    def run_one_sampling(self, data, seed, i):
        rng = np.random.default_rng(seed)

        # Generate random (but normalized) inits
        theta = normalize_with_d(rng.random((self.p + 1, self.user_groups)), self.d0)
        eta = normalize_with_d(rng.random((self.m + 1, self.item_groups)), self.d1)
        pr = normalize_with_self(rng.random((self.user_groups, self.item_groups, self.r + 1)))

        # Do the work
        for j in tqdm(range(self.iterations), position=0):
            # This is the crux of the script; please see expectation_maximization.py
            n_theta, n_eta, npr = update_coefficients(
                data=data, ratings=self.ratings, theta=theta, eta=eta, pr=pr
            )

            # Update with normalization
            theta = normalize_with_d(n_theta, self.d0)
            eta = normalize_with_d(n_eta, self.d1)
            pr = normalize_with_self(npr)

            if self.debug:
                # For debugging purposes; compute likelihood every once in a while
                if j % 50 == 0:
                    likelihood = compute_likelihood(
                        self.train, self.ratings, theta, eta, pr
                    )
                    self.logger.debug(
                        f"\nLikelihood at run {i} is {likelihood.sum():.0f}"
                    )

        likelihood = compute_likelihood(self.train, self.ratings, theta, eta, pr)

        return {
            "likelihood": likelihood,
            "pr": pr,
            "theta": theta,
            "eta": eta,
        }

    def _check_is_fitted(self):
        assert self.results is not None, "You need to fit the model before predicting."

    def _check_has_predictions(self):
        assert self.prediction_matrix is not None, (
            "You need to predict before computing the goodness of fit " "parameters."
        )

    def predict(self, data):
        """
        Use the fitted model to predict group memberships of the users of new data.
        :param data: {dataframe} of shape (n_samples, 3)
            The data to predict on. The first column has to be the user identifier, the second column the item
            identifier and the third the rating.
        :return: {ndarray}, shape (n_samples, user_groups)
            Prediction matrix. A k column, n_samples numpy array whereby each row gives the probability of each user
            belonging to each group.
        """

        self._check_is_fitted()

        test = self.data_handler.format_test_data(data)
        self.test = test

        # Get the info for all the runs
        rats = [
            compute_prod_dist(test, a["theta"], a["eta"], a["pr"])
            for a in self.results
        ]
        prs = np.array([a["pr"] for a in self.results])
        likelihoods = np.array([a["likelihood"] for a in self.results])
        thetas = np.array([a["theta"] for a in self.results])
        etas = np.array([a["eta"] for a in self.results])

        best = self.choose_best_run(rats)

        # Store the cleaned best objects
        self.theta = self.data_handler.return_theta_indices(thetas[best])
        self.eta = self.data_handler.return_eta_indices(etas[best])
        self.pr = self.data_handler.return_pr_indices(prs[best])
        self.likelihood = likelihoods[best]

        # The prediction matrix is the average of all runs
        self.prediction_matrix = np.array(rats).mean(axis=0)

        return self.prediction_matrix

    def choose_best_run(self, rats):

        # We compute the accuracy of all of them and return the index of the best
        accuracies = [self._compute_stats(a)["accuracy"] for a in rats]
        return accuracies.index(max(accuracies))

    def score(self, silent=False):
        """
        Compute the goodness of fit statistics as well as the best possible model if multiple have been fitted.
        :param silent: {boolean}.
            Do you want to shut off most of the notifications? (Mostly for internal use)
        :return: {dict}
            Dictionary with two sub-dictionaries, one for the goodness of fit statistics (stats) and another one with
            the computed objects in the model:
                stats: {dict}
                    accuracy: float
                        Ratio of observations we got right.
                    one_off_accuracy: float
                        Ratio of observations we got at most one point away from truth.
                    mae: integer
                        Mean absolute error.
                    s2: integer
                        Squared error.
                    s2pond: float
                        Weighted distance between predicted and reality.
                objects: {dict}
                    theta: ndarray (n_items, n_item_groups)
                        Item loadings on item groups.
                    eta: ndarray (n_samples, n_user_groups)
                        User loadings on user groups.
                    pr: ndarray (n_ratings, n_user_groups, n_item_groups)
                        User group loadings for item groups for each rating.
        """

        self._check_has_predictions()

        stats = self._compute_stats(self.prediction_matrix)
        stats["likelihood"] = self.likelihood

        if not silent:
            # Explain how we did
            if self.debug:
                self.logger.info(
                    f"Done {self.sampling} runs in {(datetime.now() - self.start_time).total_seconds() / 60.0:.2f} "
                    f"minutes."
                )
            self.logger.info(
                f"The final accuracy is {stats['accuracy']}, the one off accuracy is {stats['one_off_accuracy']} "
                f"and the MAE is {stats['mae']}."
            )

        return {
            "stats": stats,
            "objects": {"theta": self.theta, "eta": self.eta, "pr": self.pr},
        }

    def cv_fit(self, data, folds=5):
        """
        Fit MMSBM with 'folds' fold cross-validation.
        :param data: {dataframe} of shape (n_samples, 3)
            The input data. The first column has to be the user identifier, the second column the item identifier and
            the third the rating.
        :param folds: {integer}
            Number of folds. It must not exceed the number of different users or items.
        :return: None
        """

        items_per_fold = structure_folds(data, folds)

        temp = data
        accuracies = []
        all_results = []
        for f in range(folds):
            self.logger.info(f"Running fold {f + 1} of {folds}...")

            # Get the correct indices
            test_indices = [
                a
                for a in temp.groupby(temp.columns[0], as_index=False)
                .apply(get_n_per_group, n=items_per_fold)
                .values
            ]
            test_indices = [a for b in test_indices for a in b if str(a) != "0"]

            test = temp.loc[test_indices, :]
            train = data[~data.index.isin(test.index)]
            temp = temp[~temp.index.isin(test_indices)]

            self.fit(train, silent=True)
            self.prediction_matrix = self.predict(test)
            results = self.score(silent=True)

            # We put together the best run for each of the s samplings of each fold
            all_results.append(
                {
                    "stats": results["stats"],
                    "objects": {
                        "theta": self.theta,
                        "eta": self.eta,
                        "pr": self.pr,
                        "rat": self.prediction_matrix,
                    },
                }
            )
            accuracies.append(results["stats"]["accuracy"])

        # Now we pick the best objects
        accuracies = [a["stats"]["accuracy"] for a in all_results]
        best = accuracies.index(max(accuracies))
        self.theta = all_results[best]["objects"]["theta"]
        self.eta = all_results[best]["objects"]["eta"]
        self.pr = all_results[best]["objects"]["pr"]
        self.prediction_matrix = all_results[best]["objects"]["rat"]

        self.logger.info(f"Ran {folds} folds with accuracies {accuracies}.")
        self.logger.info(
            f"They have mean {np.mean(accuracies)} and sd {np.std(accuracies)}."
        )

        return accuracies

    def _compute_stats(self, rat):
        # How did we do?
        rat = self._compute_indicators(rat)
        # Final model quality indicators
        stats = self._compute_final_stats(rat)

        return stats

    def _compute_indicators(self, rat):

        rat = pd.DataFrame(rat)
        rat["pred"] = np.argmax(rat.values, axis=1)

        # Add the real results
        rat = rat.assign(real=pd.Series(self.test[:, 2]))

        # Remove observations without predictions
        rat = rat.loc[rat.iloc[:, : len(self.ratings)].sum(axis=1) != 0, :]

        # Check the ones we got right
        rat["true"] = np.where(rat["pred"] == rat["real"], 1, 0)

        # Check the ones we got almost right
        rat["almost"] = np.where(abs(rat["pred"] - rat["real"]) <= 1, 1, 0)

        # squared error (which is not squared error but ok)
        rat["s2"] = abs(rat["pred"] - rat["real"])

        # Same but weighed
        # Note that we are assuming that weights are the first R columns
        rat["pred_pond"] = [
            self._weighting(a, self.ratings)
            for a in rat.iloc[:, : len(self.ratings)].values
        ]
        rat["true_pond"] = np.where(rat["real"] == round(rat["pred_pond"]), 1, 0)
        rat["s2pond"] = abs(rat["pred_pond"] - rat["real"])

        return rat

    @staticmethod
    def _compute_final_stats(rat):
        # Final model quality indicators
        accuracy = rat["true"].sum() / rat.shape[0]
        one_off_accuracy = rat["almost"].sum() / rat.shape[0]
        mae = 1 - rat["true_pond"].sum() / rat.shape[0]

        # Errors
        s2 = rat["s2"].sum()
        s2pond = rat["s2pond"].sum()

        return {
            "accuracy": accuracy,
            "one_off_accuracy": one_off_accuracy,
            "mae": mae,
            "s2": s2,
            "s2pond": s2pond,
        }

    @staticmethod
    def _weighting(x, ratings):
        return sum([a * b for (a, b) in zip(x, ratings)])
