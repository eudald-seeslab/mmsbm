import multiprocessing
from datetime import datetime
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data_handler import DataHandler
from expectation_maximization import ExpectationMaximization
from helpers import structure_folds, get_n_per_group

from logger import setup_logger


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

    backend: str, default="auto"
        The backend to use for ExpectationMaximization.

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
        backend="auto",
    ):
        self.start_time = datetime.now()
        self.user_groups = user_groups
        self.item_groups = item_groups
        self.iterations = iterations
        self.sampling = sampling
        self.debug = debug
        self.backend = backend

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
            norm_factors=self._normalization_factors,
            backend=self.backend,
            debug=self.debug
        )

    def fit(self, data, silent=False):
        """Fits the MMSBM using multiple EM runs.

         Performs 'sampling' parallel runs of the EM algorithm and keeps track
         of all results for later model selection.

         Args:
             data: DataFrame with columns [users, items, ratings]
             silent: If True, suppresses progress output

         See Also:
             cv_fit: For cross-validated model fitting
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
        """Executes an EM optimization run with random initialization.

        Performs one complete run of the Expectation-Maximization algorithm:
        1. Initializes model parameters randomly using the provided seed
        2. Iteratively updates parameters using the EM algorithm
        3. Monitors convergence through likelihood if in debug mode
        4. Returns final parameters and likelihood

        Args:
            data: Array of shape (n_samples, 3) with [user_idx, item_idx, rating]
                Training data for this sampling run.
            seed: int or numpy.random.SeedSequence
                Random seed for parameter initialization.
                Ensures reproducibility across runs.
            i: int
                Index of current sampling run.
                Used for logging and progress tracking.

        Returns:
            dict: Final model state containing:
                - likelihood: Final log-likelihood of the model
                - pr: Rating probabilities, shape (n_user_groups, n_item_groups, n_ratings)
                - theta: User group memberships, shape (n_users, n_user_groups)
                - eta: Item group memberships, shape (n_items, n_item_groups)

        Notes:
            - Parameters are initialized randomly but normalized to valid probabilities
            - Uses vectorized EM implementation for efficiency
            - If debug=True, prints likelihood every 50 iterations
            - Each run is independent and can converge to different local optima

        See Also:
            fit: For running multiple sampling runs
            ExpectationMaximization.update_coefficients: For EM implementation details
        """

        rng = np.random.default_rng(seed)

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
        """Predicts ratings for new user-item pairs.

        Uses the best model from all sampling runs to make predictions.

        Args:
            data: DataFrame with columns [users, items, ratings]

        Returns:
            Array with rating probabilities for each possible rating value
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

    def score(self, silent=False):
        """Computes model performance metrics and returns fitted parameters.

        Evaluates the model's predictive performance using multiple metrics
        and returns both the evaluation statistics and the fitted model parameters.

        Args:
            silent: If True, suppresses logging output. Defaults to False.

        Returns:
            dict: Contains two sub-dictionaries:
                stats: Model performance metrics
                    - accuracy: Percentage of exactly correct predictions
                    - one_off_accuracy: Percentage of predictions off by at most 1
                    - mae: Mean Absolute Error of predictions
                    - s2: Sum of squared differences between predicted and actual ratings
                    - s2pond: Weighted squared error considering rating probabilities
                objects: Fitted model parameters
                    - theta: User group memberships, shape (n_users, n_user_groups)
                    - eta: Item group memberships, shape (n_items, n_item_groups)
                    - pr: Rating probabilities per group pair, shape (n_user_groups, n_item_groups, n_ratings)

        Example:
            >>> results = model.score()
            >>> print(f"Model accuracy: {results['stats']['accuracy']:.3f}")
            >>> print(f"User memberships:\n{results['objects']['theta']}")

        See Also:
            predict: For making predictions on new data
            cv_fit: For cross-validated model evaluation
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
        """Fits the model using k-fold cross-validation.

        Splits the data into k folds and performs multiple training runs, each time
        holding out one fold for validation. This provides a more robust estimate
        of model performance and helps avoid overfitting.

        The function:
        1. Splits users into k groups
        2. For each fold:
            - Uses k-1 groups for training
            - Tests on the remaining group
            - Runs multiple sampling iterations
        3. Returns accuracies for each fold

        Args:
            data: DataFrame with user, item, and rating columns
                The input data for model fitting and validation.
                Shape: (n_samples, 3)
            folds: int, default=5
                Number of cross-validation folds.
                Must not exceed the number of unique users or items.

        Returns:
            list: Prediction accuracies for each fold
                Can be used to compute mean performance and confidence intervals.

        Example:
            >>> accuracies = model.cv_fit(data, folds=5)
            >>> print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

        Raises:
            AssertionError: If number of folds exceeds number of unique items.

        Notes:
            - Each fold preserves the user rating distribution
            - Uses parallel processing for sampling runs
            - Stores best model parameters from all folds

        See Also:
            fit: For simple model fitting without cross-validation
            score: For evaluating model performance
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

    def choose_best_run(self, rats):

        # We compute the accuracy of all of them and return the index of the best
        accuracies = [self._compute_stats(a)["accuracy"] for a in rats]
        return accuracies.index(max(accuracies))

    def _compute_stats(self, rat):
        # How did we do?
        rat = self._compute_indicators(rat)
        # Final model quality indicators
        stats = self._compute_final_stats(rat)

        return stats

    def _compute_indicators(self, rat):
        """Compute boolean/error indicators for evaluation in pure NumPy.

        Parameters
        ----------
        rat : ndarray of shape (N, R)
            Predicted rating probability distribution for each observation.

        Returns
        -------
        dict of str -> ndarray
            Arrays needed by _compute_final_stats.
        """

        # Predicted (most probable) rating index
        pred = np.argmax(rat, axis=1)
        real = self.test[:, 2]

        # Filter observations without predictions (all-zero probability rows)
        mask = rat.sum(axis=1) != 0
        if not np.all(mask):
            pred = pred[mask]
            real = real[mask]
            rat = rat[mask]

        true = (pred == real).astype(int)
        almost = (np.abs(pred - real) <= 1).astype(int)
        s2 = np.abs(pred - real)

        # Weighted quantities using the actual probability distribution
        pred_pond = rat @ self.ratings  # dot per row
        true_pond = (real == np.round(pred_pond)).astype(int)
        s2pond = np.abs(pred_pond - real)

        return {
            "true": true,
            "almost": almost,
            "s2": s2,
            "true_pond": true_pond,
            "s2pond": s2pond,
        }

    def _compute_final_stats(self, rat):
        n = len(rat["true"])

        return {
            "accuracy": rat["true"].sum() / n,
            "one_off_accuracy": rat["almost"].sum() / n,
            "mae": 1 - rat["true_pond"].sum() / n,
            "s2": rat["s2"].sum(),
            "s2pond": rat["s2pond"].sum(),
        }

    def compute_likelihood(self, data, theta, eta, pr):
        """Optimized version using pre-computed arrays and handling zeros"""
        omegas = self.em.compute_omegas(data, theta, eta, pr)
        sum_omega = np.zeros(self._dims['n_samples'])
        np.sum(omegas, axis=(1, 2), out=sum_omega)

        # Small epsilon to avoid log(0)
        epsilon = np.finfo(float).eps
        safe_omegas = np.maximum(omegas, epsilon)
        safe_sums = np.maximum(sum_omega, epsilon)

        return np.sum(safe_omegas * np.log(safe_omegas) -
                      safe_omegas * np.log(safe_sums[:, np.newaxis, np.newaxis]))
