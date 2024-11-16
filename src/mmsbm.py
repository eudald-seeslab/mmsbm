import multiprocessing
from datetime import datetime
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data_handler import DataHandler
from expectation_maximization import ExpectationMaximization
from helpers import structure_folds, get_n_per_group

from src.logger import setup_logger


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
    rng = None

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
        self.rng = np.random.default_rng(seed)
        # Initiate the child random states
        ss = self.rng.bit_generator._seed_seq
        self.child_states = ss.spawn(sampling)

        self.logger = setup_logger("MMSBM")

        self._normalization_factors = None
        self._user_indices = None
        self._item_indices = None

    def _prepare_objects(self, train):

        self.ratings = sorted(set(train[:, 2]))
        self.r = max(self.ratings)
        self.p = int(train[:, 0].max())
        self.m = int(train[:, 1].max())

        self.d0 = {a: list(train[train[:, 0] == a, 1]) for a in set(train[:, 0])}
        self.d1 = {a: list(train[train[:, 1] == a, 0]) for a in set(train[:, 1])}

        self.train = train

        # Pre-compute normalization factors
        self._normalization_factors = {
            'user': np.array([np.repeat(max(len(a), 1), self.user_groups)
                              for a in self.d0.values()]),
            'item': np.array([np.repeat(max(len(a), 1), self.item_groups)
                              for a in self.d1.values()])
        }

        # Pre-compute indices for update_coefficients
        self._user_indices = [
            np.where(train[:, 0] == a)[0] for a in range(self.p + 1)
        ]
        self._item_indices = [
            np.where(train[:, 1] == a)[0] for a in range(self.m + 1)
        ]
        self._rating_indices = [
            np.where(train[:, 2] == a)[0] for a in self.ratings
        ]

        # Pre-compute dimensions
        self._dims = {
            'n_samples': len(train),
            'n_user_groups': self.user_groups,
            'n_item_groups': self.item_groups,
            'n_ratings': len(self.ratings)
        }

        # Pre-allocate arrays for results
        self._omegas = np.zeros((self._dims['n_samples'],
                                 self._dims['n_user_groups'],
                                 self._dims['n_item_groups']))

        # Initialize the expectation-maximization algorithm
        self.em = ExpectationMaximization(
            dims=self._dims,
            user_indices=self._user_indices,
            item_indices=self._item_indices,
            rating_indices=self._rating_indices,
            norm_factors=self._normalization_factors
        )

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

    def normalize_with_d(self, df, type_):
        """Optimized version using pre-computed normalization factors"""
        return df / self._normalization_factors[type_]

    def compute_omegas(self, data, theta, eta, pr):
        """Optimized version using pre-computed dimensions"""
        user_indices = data[:, 0]
        item_indices = data[:, 1]
        rating_indices = data[:, 2]

        self._omegas[:] = (theta[user_indices][:, :, np.newaxis] *
                           eta[item_indices][:, np.newaxis, :] *
                           np.moveaxis(pr[:, :, rating_indices], -1, 0))

        return self._omegas

    def update_coefficients(self, data, theta, eta, pr):
        """Optimized version using pre-computed dimensions and cached indices"""
        omegas = self.compute_omegas(data, theta, eta, pr)
        sum_omega = np.zeros(self._dims['n_samples'])
        np.sum(omegas, axis=(1, 2), out=sum_omega)

        increments = np.divide(omegas, sum_omega[:, np.newaxis, np.newaxis])

        n_theta = np.zeros((self.p + 1, self._dims['n_user_groups']))
        n_eta = np.zeros((self.m + 1, self._dims['n_item_groups']))
        n_pr = np.zeros((self._dims['n_user_groups'],
                         self._dims['n_item_groups'],
                         self._dims['n_ratings']))

        # Usar les arrays pre-allocades
        for idx, indices in enumerate(self._user_indices):
            n_theta[idx] = increments[indices].sum(axis=(0, -1))

        for idx, indices in enumerate(self._item_indices):
            n_eta[idx] = increments[indices].sum(axis=(0, 1))

        for idx, indices in enumerate(self._rating_indices):
            n_pr[:, :, idx] = increments[indices].sum(axis=0)

        return n_theta, n_eta, n_pr

    def run_one_sampling(self, data, seed, i):
        rng = np.random.default_rng(seed)

        # Inicialitzacions amb l'objecte EM
        theta = self.em.normalize_with_d(
            rng.random((self.p + 1, self._dims['n_user_groups'])), 'user')
        eta = self.em.normalize_with_d(
            rng.random((self.m + 1, self._dims['n_item_groups'])), 'item')
        pr = self.em.normalize_with_self(
            rng.random((self._dims['n_user_groups'],
                        self._dims['n_item_groups'],
                        self._dims['n_ratings'])))

        for j in tqdm(range(self.iterations)):
            n_theta, n_eta, npr = self.em.update_coefficients(
                data=data, theta=theta, eta=eta, pr=pr
            )

            theta = self.em.normalize_with_d(n_theta, 'user')
            eta = self.em.normalize_with_d(n_eta, 'item')
            pr = self.em.normalize_with_self(npr)

            if self.debug and j % 50 == 0:
                likelihood = self.em.compute_likelihood(self.train, theta, eta, pr)
                self.logger.debug(f"\nLikelihood at run {i} is {likelihood.sum():.0f}")

        likelihood = self.em.compute_likelihood(self.train, theta, eta, pr)

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
            self.em.compute_prod_dist(test, a["theta"], a["eta"], a["pr"])
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
            self.logger.debug(
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
                .apply(get_n_per_group, n=items_per_fold, rng=self.rng)
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
        rat['real'] = self.test[:, 2]

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
        rat["pred_pond"] = np.dot(rat.iloc[:, :len(self.ratings)].values, self.ratings)
        rat["true_pond"] = np.where(rat["real"] == np.round(rat["pred_pond"]), 1, 0)
        rat["s2pond"] = np.abs(rat["pred_pond"] - rat["real"])

        return rat

    @staticmethod
    def _compute_final_stats(rat):
        n = rat.shape[0]

        return {
            "accuracy": rat["true"].sum() / n,
            "one_off_accuracy": rat["almost"].sum() / n,
            "mae": 1 - rat["true_pond"].sum() / n,
            "s2": rat["s2"].sum(),
            "s2pond": rat["s2pond"].sum(),
        }

    def compute_likelihood(self, data, theta, eta, pr):
        """Optimized version using pre-computed arrays and handling zeros"""
        omegas = self.compute_omegas(data, theta, eta, pr)
        sum_omega = np.zeros(self._dims['n_samples'])
        np.sum(omegas, axis=(1, 2), out=sum_omega)

        # Small epsilon to avoid log(0)
        epsilon = np.finfo(float).eps
        safe_omegas = np.maximum(omegas, epsilon)
        safe_sums = np.maximum(sum_omega, epsilon)

        return np.sum(safe_omegas * np.log(safe_omegas) -
                      safe_omegas * np.log(safe_sums[:, np.newaxis, np.newaxis]))
