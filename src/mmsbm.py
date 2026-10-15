import multiprocessing
from datetime import datetime
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data_handler import DataHandler
from expectation_maximization import ExpectationMaximization
from helpers import assign_folds, drop_unseen

from logger import setup_logger

__version__ = "1.3.0"


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
        bigger than 0 adds to robustness of the model. Predictions average all runs; the run with the highest
        training log-likelihood is the one exposed as ``theta``, ``eta``, ``pr`` and ``likelihood``.

    seed: int or None, default=None
        Seed for reproducibility. With ``None`` every fit starts from fresh entropy and the results are not
        reproducible.

    debug: bool, default=False
        Print the backend in use and log the training log-likelihood every 50 iterations.

    backend: str, default="auto"
        The backend used by ExpectationMaximization: ``"numpy"``, ``"numba"``, ``"cupy"`` or ``"auto"`` (the first
        available of cupy, numba, numpy).

    degree_corrected: bool, default=False
        Fit the degree-corrected variant of the model. The standard MMSBM only models the rating given that a user
        has rated an item. The degree-corrected model (the mixed-membership analogue of the degree-corrected
        stochastic block model) also models *which* user-item pairs are observed, treating every (user, item, rating)
        cell as a Poisson count with rate Σ_kℓ θ_uk η_iℓ p_kℓ(r). Memberships are then unnormalised: each row of
        theta sums to the number of ratings of that user, each row of eta to the number of ratings of that item, so
        degree heterogeneity is absorbed by the parameters instead of driving the groups. The EM kernels are shared
        with the standard model; only the M-step normalisation, the likelihood and the prediction step differ.

    chunk_size: int or None, default=None
        Rows per block of the E-step with the numpy and cupy backends, which materialise the responsibility tensor
        (rows × user_groups × item_groups, in float64) of the rows they process. ``None`` chooses as many rows as
        keep that tensor under 256 MiB, so the memory of an iteration is bounded whatever the number of ratings.
        Data shorter than one block is processed exactly as before; longer data sums the partial numerators of
        the blocks, which can change results in the last bits. The numba M-step never materialises the tensor and
        ignores this parameter.

    log_file: str or None, default="mmsbm.log"
        File that receives a copy of the log messages, relative to the working directory. ``None`` logs to the
        console only.

    Attributes
    ---------
    results: list of dict or None
        One entry per sampling run after :meth:`fit`, with the keys ``likelihood``, ``theta``, ``eta`` and ``pr``
        (see :meth:`run_one_sampling`). The goodness-of-fit statistics are returned by :meth:`score`.

    theta, eta, pr: DataFrame, DataFrame, dict of DataFrame or None
        Parameters of the best run with the original user, item and rating labels, set by :meth:`predict`.

    prediction_matrix: ndarray or None
        Rating probabilities for the rows passed to the last :meth:`predict`, averaged over runs.

    selected_groups: tuple or None
        (user_groups, item_groups) selected by the last call to :meth:`select_groups`.

    """

    data_handler = None
    results = None
    train = None
    test = None
    _d0 = None
    _d1 = None
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
        degree_corrected=False,
        chunk_size=None,
        log_file="mmsbm.log",
    ):
        self.start_time = datetime.now()
        self.user_groups = user_groups
        self.item_groups = item_groups
        self.iterations = iterations
        self.sampling = sampling
        self.debug = debug
        self.backend = backend
        self.degree_corrected = degree_corrected
        self.chunk_size = chunk_size
        self.log_file = log_file
        self.selected_groups = None
        self._show_progress = True

        # Initiate the general random state
        self.rng = np.random.default_rng(seed)
        # Initiate the child random states
        ss = self.rng.bit_generator._seed_seq
        self.child_states = ss.spawn(sampling)

        self.logger = setup_logger("MMSBM", log_file=log_file)

        self._normalization_factors = None
        self._user_indices = None
        self._item_indices = None

    def _prepare_objects(self, train):

        self.ratings = np.unique(train[:, 2]).tolist()
        self.r = max(self.ratings)
        self.p = int(train[:, 0].max())
        self.m = int(train[:, 1].max())

        self.train = train
        self._d0 = None
        self._d1 = None

        # Pre-compute normalization factors: the degree of every user and
        # item, repeated along the group axis so that it divides theta and
        # eta row-wise.  Linear in the number of ratings.
        user_degrees = np.bincount(train[:, 0], minlength=self.p + 1)
        item_degrees = np.bincount(train[:, 1], minlength=self.m + 1)
        self._normalization_factors = {
            'user': np.repeat(np.maximum(user_degrees, 1)[:, None], self.user_groups, axis=1),
            'item': np.repeat(np.maximum(item_degrees, 1)[:, None], self.item_groups, axis=1),
        }

        # Pre-compute dimensions
        self._dims = {
            'n_samples': len(train),
            'n_user_groups': self.user_groups,
            'n_item_groups': self.item_groups,
            'n_ratings': len(self.ratings)
        }

        # Initialize the expectation-maximization algorithm
        self.em = ExpectationMaximization(
            dims=self._dims,
            norm_factors=self._normalization_factors,
            backend=self.backend,
            debug=self.debug,
            chunk_size=self.chunk_size,
        )

    @staticmethod
    def _neighbours(keys, values):
        """``{key: [values with that key]}`` for two aligned index arrays."""
        order = np.argsort(keys, kind="stable")
        unique, starts = np.unique(keys[order], return_index=True)
        return {k: list(v) for k, v in zip(unique, np.split(values[order], starts[1:]))}

    @property
    def d0(self):
        """Items rated by each user, ``{user_idx: [item_idx, ...]}``.

        Built on first access from the training data (it is not needed by the
        EM algorithm and is kept for backwards compatibility).
        """
        assert self.train is not None, "You need to fit the model first."
        if self._d0 is None:
            self._d0 = self._neighbours(self.train[:, 0], self.train[:, 1])
        return self._d0

    @property
    def d1(self):
        """Users that rated each item, ``{item_idx: [user_idx, ...]}``.

        Built on first access from the training data (it is not needed by the
        EM algorithm and is kept for backwards compatibility).
        """
        assert self.train is not None, "You need to fit the model first."
        if self._d1 is None:
            self._d1 = self._neighbours(self.train[:, 1], self.train[:, 0])
        return self._d1

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

        if self.sampling == 1:
            # A single run needs no pool of worker processes: run it here,
            # with the same seed it would have received in a worker.
            self.results = [self.run_one_sampling(train, self.child_states[0], 0)]
            return

        # Ensure tqdm works nicely with multiprocessing
        try:
            from multiprocessing import RLock
            tqdm.set_lock(RLock())
        except Exception:
            # If we cannot set the lock (e.g. on certain platforms), just continue
            pass

        # Always use 'spawn' for cross-platform safety and to avoid
        # fork() deadlock issues in multi-threaded processes
        ctx = multiprocessing.get_context('spawn')

        with ctx.Pool(processes=self.sampling) as pool:
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
                - likelihood: Log-likelihood of the training data under the final
                  parameters, Σ log p(r | u, i) (the Poisson log-likelihood when
                  ``degree_corrected=True``)
                - pr: Rating probabilities, shape (n_user_groups, n_item_groups, n_ratings)
                - theta: User group memberships, shape (n_users, n_user_groups)
                - eta: Item group memberships, shape (n_items, n_item_groups)

            With ``degree_corrected=True`` the rows of theta and eta sum to the
            user and item degrees, and pr holds the Poisson rates per pair of
            half-edges rather than probabilities over ratings.

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

        if self.degree_corrected:
            # Break the symmetry between groups with a random 10:1 affinity
            # pattern; a uniform affinity is a stationary point of the Poisson
            # likelihood from which the EM cannot escape (see
            # ExpectationMaximization.initial_affinity).  The draws above are
            # the same as in the standard model, so only the extra draw differs.
            pr = pr * self.em.initial_affinity(
                rng, self._dims['n_user_groups'], self._dims['n_item_groups'])[:, :, None]

        # With the cupy backend the data and the parameters live on the GPU
        # for the whole run and only the final parameters come back; with the
        # other backends these calls return their argument.
        data = self.em.to_device(data)
        theta = self.em.to_device(theta)
        eta = self.em.to_device(eta)
        pr = self.em.to_device(pr)

        if self.em._backend == 'numba' and self.sampling > 1:
            # Each run is a separate process: share the cores between them
            # instead of letting every process use all of them.
            import numba
            numba.set_num_threads(max(1, numba.config.NUMBA_NUM_THREADS // self.sampling))

        # Configure tqdm to avoid overlapping bars when using multiprocessing
        tqdm_kwargs = {
            "position": i,
            "leave": False,
            "ascii": False,
            "disable": not self._show_progress,
        }

        # The degree-corrected model shares the E-step and the accumulation
        # of the numerators with the standard model; only the M-step
        # normalisation and the likelihood differ.
        degree_corrected = self.degree_corrected
        likelihood_fn = (self.em.poisson_log_likelihood if degree_corrected
                         else self.em.compute_likelihood)

        for j in tqdm(range(self.iterations), **tqdm_kwargs):
            n_theta, n_eta, npr = self.em.update_coefficients(
                data=data, theta=theta, eta=eta, pr=pr
            )

            if degree_corrected:
                theta, eta, pr = self.em.m_step_degree_corrected(n_theta, n_eta, npr)
            else:
                theta = self.em.normalize_with_d(n_theta, 'user')
                eta = self.em.normalize_with_d(n_eta, 'item')
                pr = self.em.normalize_with_self(npr)

            if self.debug and j % 50 == 0:
                likelihood = self.em.to_host(likelihood_fn(data, theta, eta, pr))
                self.logger.debug(f"\nLog-likelihood at run {i} is {likelihood:.0f}")

        likelihood = likelihood_fn(data, theta, eta, pr)

        theta, eta, pr, likelihood = (self.em.to_host(a) for a in (theta, eta, pr, likelihood))

        # If using GPU, explicitly free memory in the worker process before returning
        if self.em._backend == 'cupy':
            import cupy as cp
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()

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

        The returned probabilities are the average over all sampling runs.
        The parameters exposed afterwards as ``theta``, ``eta``, ``pr`` and
        ``likelihood`` are those of the run with the highest training
        log-likelihood (see :meth:`choose_best_run`).

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
        if self.degree_corrected:
            # prod_dist returns Poisson rates λ_uir; condition on the pair
            # being observed to get p(r | u, i).
            rats = [self.em.normalize_rows(a) for a in rats]

        best = self.choose_best_run()

        # Store the cleaned best objects
        self.theta = self.data_handler.return_theta_indices(self.results[best]["theta"])
        self.eta = self.data_handler.return_eta_indices(self.results[best]["eta"])
        self.pr = self.data_handler.return_pr_indices(self.results[best]["pr"])
        self.likelihood = self.results[best]["likelihood"]

        # The prediction matrix is the average of all runs
        self.prediction_matrix = np.array(rats).mean(axis=0)

        return self.prediction_matrix

    def predict_links(self, data):
        """Predicts the probability that each (user, item) pair is observed.

        Only available for the degree-corrected model, which models which
        pairs are rated. Under the Poisson model the probability that user u
        rates item i at all is

            P(A_ui > 0) = 1 - exp(-Σ_r λ_uir),

        averaged over all sampling runs, as in :meth:`predict`.

        Args:
            data: DataFrame with columns [users, items]; a ratings column, if
                present, is ignored. Pairs whose user or item was not in the
                training set are dropped with a warning.

        Returns:
            Array with one probability per (kept) row of ``data``.
        """

        self._check_is_fitted()
        assert self.degree_corrected, (
            "Link prediction is only defined for the degree-corrected model; "
            "fit with degree_corrected=True."
        )

        pairs = self.data_handler.format_pairs(data)
        probabilities = [
            1.0 - np.exp(-self.em.compute_prod_dist(pairs, a["theta"], a["eta"], a["pr"]).sum(axis=1))
            for a in self.results
        ]
        return np.mean(probabilities, axis=0)

    def held_out_log_likelihood(self, smoothing=0.0):
        """Mean log-probability of the observed held-out ratings.

        Computes the average over the rows passed to :meth:`predict` of
        log p(r_ui | u, i), using the prediction matrix (the average over
        sampling runs). It is defined identically for the standard and the
        degree-corrected model, so it can be used to compare them and to choose
        the number of groups (see :meth:`select_groups`).

        Args:
            smoothing: float in [0, 1], default=0.0
                Weight of a uniform distribution over the ratings mixed into
                the predicted distribution before taking logs,
                p' = (1 - smoothing) p + smoothing / n_ratings. Maximum-likelihood
                fits can assign a vanishing probability to a rating that is rare
                for a pair of groups, and a single such held-out rating would
                otherwise dominate the mean. With 0 the raw predictions are
                used, floored at machine precision.

        Returns:
            float: Mean held-out log-likelihood per rating (non-positive).
        """

        self._check_has_predictions()
        assert 0 <= smoothing <= 1, "smoothing must be between 0 and 1."

        real = self.test[:, 2]
        probabilities = self.prediction_matrix[np.arange(len(real)), real]
        if smoothing:
            probabilities = (1 - smoothing) * probabilities + smoothing / self.prediction_matrix.shape[1]
        eps = np.finfo(float).eps
        return float(np.mean(np.log(np.maximum(probabilities, eps))))

    def score(self, silent=False):
        """Computes model performance metrics and returns fitted parameters.

        Evaluates the model's predictive performance using multiple metrics
        and returns both the evaluation statistics and the fitted model parameters.

        .. versionchanged:: 1.2.0
            ``mae`` is now the mean absolute error of the most probable rating.
            Before it was one minus ``expected_rating_accuracy``.

        Two point predictions are evaluated for every held-out rating: the
        *most probable* rating, argmax_r p(r | u, i), and the *expected*
        rating, Σ_r r p(r | u, i). Errors are measured in rating units when
        every rating label is a number (so that half-star ratings or ratings
        from 1 to 10 are handled correctly) and in positions of the sorted
        list of distinct ratings otherwise. Rows whose predicted distribution
        is all zeros are left out.

        Args:
            silent: If True, suppresses logging output. Defaults to False.

        Returns:
            dict: Contains two sub-dictionaries:
                stats: Model performance metrics
                    - accuracy: Fraction of ratings equal to the most probable rating
                    - one_off_accuracy: Fraction of ratings within one unit of the most probable rating
                    - mae: Mean absolute error of the most probable rating
                    - expected_rating_accuracy: Fraction of ratings equal to the rating level nearest to the
                      expected rating
                    - expected_rating_mae: Mean absolute error of the expected rating
                    - s2: Sum of absolute errors of the most probable rating (``mae`` times the number of
                      predictions; kept for backwards compatibility)
                    - s2pond: Sum of absolute errors of the expected rating (``expected_rating_mae`` times the
                      number of predictions; kept for backwards compatibility)
                    - likelihood: Training log-likelihood of the run exposed in ``objects``
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
        1. Assigns every rating to a fold with :func:`helpers.assign_folds`:
           the ratings of each user are dealt round-robin across the folds,
           so every user with at least two ratings is in the training part of
           every fold and the folds have (almost) the same size
        2. For each fold:
            - Trains on the other k-1 folds
            - Tests on the fold, leaving out the ratings whose user or item
              does not appear in the training folds
            - Runs multiple sampling iterations
        3. Returns accuracies for each fold

        .. versionchanged:: 1.2.0
            Before 1.2.0 every fold took, from every user, a fixed number of
            ratings equal to the number of distinct items divided by
            ``folds``. On sparse data (users with fewer ratings than that)
            the first fold took all the ratings and the remaining folds were
            empty; the test parts also overlapped with later training parts.
            The splits, and therefore the returned accuracies, differ from
            earlier versions for the same seed.

        Args:
            data: DataFrame with user, item, and rating columns
                The input data for model fitting and validation.
                Shape: (n_samples, 3)
            folds: int, default=5
                Number of cross-validation folds, between 2 and the number of
                ratings.

        Returns:
            list: Prediction accuracies for each fold
                Can be used to compute mean performance and confidence intervals.

        Example:
            >>> accuracies = model.cv_fit(data, folds=5)
            >>> print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

        Raises:
            AssertionError: If ``folds`` is smaller than 2 or larger than the number of ratings.

        Notes:
            - Uses parallel processing for sampling runs
            - Stores the parameters of the fold with the highest accuracy

        See Also:
            fit: For simple model fitting without cross-validation
            score: For evaluating model performance
            select_groups: For choosing the number of groups by cross-validation
        """

        assert 2 <= folds <= len(data), (
            f"folds must be between 2 and the number of ratings ({len(data)})."
        )
        fold_ids = assign_folds(data, folds, self.rng)

        accuracies = []
        all_results = []
        for f in range(folds):
            self.logger.info(f"Running fold {f + 1} of {folds}...")

            train = data[fold_ids != f]
            test = drop_unseen(train, data[fold_ids == f])

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

    def select_groups(self, data, user_groups, item_groups, folds=5,
                      criterion="log_likelihood", smoothing=0.05):
        """Chooses the number of user and item groups by cross-validation.

        For every combination of ``user_groups`` x ``item_groups`` a model
        with this instance's settings (iterations, sampling, backend,
        degree_corrected) is fitted on k-1 folds and evaluated on the held-out
        fold with :meth:`held_out_log_likelihood` and :meth:`score`. Folds are
        built with :func:`helpers.assign_folds`, which spreads each user's
        ratings across folds. Ratings whose user or item is absent from the
        training folds are left out of the evaluation.

        The held-out log-likelihood is preferred over the training likelihood
        with a BIC-type penalty because the number of membership parameters
        grows with the number of users and items, which is outside the regime
        where such penalties are justified. Because the fits are maximum
        likelihood, the predicted distributions are smoothed towards the
        uniform distribution before taking logs (see ``smoothing``); otherwise
        a few held-out ratings predicted with near-zero probability dominate
        the criterion and favour too few groups.

        Args:
            data: DataFrame with columns [users, items, ratings]
            user_groups: Iterable of candidate numbers of user groups.
            item_groups: Iterable of candidate numbers of item groups.
            folds: int, default=5
                Number of cross-validation folds.
            criterion: str, default="log_likelihood"
                Column used to pick the best combination: ``"log_likelihood"``
                or ``"accuracy"``.
            smoothing: float, default=0.05
                Passed to :meth:`held_out_log_likelihood`.

        Returns:
            DataFrame: One row per combination with columns ``user_groups``,
            ``item_groups``, ``log_likelihood`` (mean over folds of the mean
            smoothed held-out log-likelihood per rating), ``log_likelihood_sd``,
            ``accuracy`` and ``accuracy_sd``. The best combination according
            to ``criterion`` is also stored in ``self.selected_groups``.

        Example:
            >>> grid = model.select_groups(data, user_groups=[2, 3, 4], item_groups=[2, 4])
            >>> best = MMSBM(*model.selected_groups, degree_corrected=model.degree_corrected)
        """

        assert criterion in ("log_likelihood", "accuracy"), (
            "criterion must be 'log_likelihood' or 'accuracy'."
        )
        user_groups = list(user_groups)
        item_groups = list(item_groups)
        fold_ids = assign_folds(data, folds, self.rng)

        rows = []
        for k in user_groups:
            for l in item_groups:
                self.logger.info(f"Evaluating {k} user groups and {l} item groups on {folds} folds...")
                log_likelihoods, accuracies = [], []
                for f in range(folds):
                    train = data[fold_ids != f]
                    test = drop_unseen(train, data[fold_ids == f])

                    model = MMSBM(
                        k,
                        l,
                        iterations=self.iterations,
                        sampling=self.sampling,
                        seed=int(self.rng.integers(2**32)),
                        backend=self.backend,
                        degree_corrected=self.degree_corrected,
                        chunk_size=self.chunk_size,
                        log_file=self.log_file,
                    )
                    model._show_progress = False
                    model.fit(train, silent=True)
                    model.predict(test)

                    log_likelihoods.append(model.held_out_log_likelihood(smoothing))
                    accuracies.append(model.score(silent=True)["stats"]["accuracy"])

                rows.append(
                    {
                        "user_groups": k,
                        "item_groups": l,
                        "log_likelihood": np.mean(log_likelihoods),
                        "log_likelihood_sd": np.std(log_likelihoods),
                        "accuracy": np.mean(accuracies),
                        "accuracy_sd": np.std(accuracies),
                    }
                )

        results = pd.DataFrame(rows)
        best = results[criterion].idxmax()
        self.selected_groups = (
            int(results.loc[best, "user_groups"]),
            int(results.loc[best, "item_groups"]),
        )
        self.logger.info(
            f"Best held-out {criterion} with {self.selected_groups[0]} user groups and "
            f"{self.selected_groups[1]} item groups."
        )

        return results

    def choose_best_run(self, rats=None):
        """Index of the sampling run with the highest training log-likelihood.

        .. versionchanged:: 1.2.0
            Runs used to be ranked by their accuracy on the data passed to
            :meth:`predict`, which selected on the test set. They are now
            ranked by the log-likelihood of the training data, as the
            documentation always stated. With ``sampling=1`` nothing changes.

        Args:
            rats: Ignored; kept for backwards compatibility.

        Returns:
            int: Position in ``self.results`` of the best run (the first one in
            case of a tie).
        """

        self._check_is_fitted()
        likelihoods = [float(a["likelihood"]) for a in self.results]
        return int(np.argmax(likelihoods))

    def _compute_stats(self, rat):
        # How did we do?
        rat = self._compute_indicators(rat)
        # Final model quality indicators
        stats = self._compute_final_stats(rat)

        return stats

    def _rating_values(self, n_ratings):
        """Value of every rating level, in the model's index order.

        The levels are the keys of the data handler's ratings dictionary. When
        all of them parse as numbers, errors are measured in rating units;
        otherwise (or before the data handler exists) every level is worth
        its position in the sorted list of distinct ratings, which is what
        versions before 1.2.0 always used.
        """
        handler = self.data_handler
        if handler is not None and handler.ratings_dict is not None:
            labels = [None] * n_ratings
            for label, index in handler.ratings_dict.items():
                if index < n_ratings:
                    labels[index] = label
            try:
                return np.array([float(label) for label in labels])
            except (TypeError, ValueError):
                pass
        return np.arange(n_ratings, dtype=float)

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

        values = self._rating_values(rat.shape[1])
        real_value = values[real]

        # Most probable rating
        true = (pred == real).astype(int)
        error = np.abs(values[pred] - real_value)
        almost = (error <= 1 + 1e-9).astype(int)

        # Expected rating under the predicted distribution, and the rating
        # level nearest to it (the lower one in case of a tie)
        expected = rat @ values
        nearest = values[np.argmin(np.abs(expected[:, np.newaxis] - values[np.newaxis, :]), axis=1)]
        true_pond = (nearest == real_value).astype(int)
        s2pond = np.abs(expected - real_value)

        return {
            "true": true,
            "almost": almost,
            "s2": error,
            "true_pond": true_pond,
            "s2pond": s2pond,
        }

    def _compute_final_stats(self, rat):
        n = len(rat["true"])

        return {
            "accuracy": rat["true"].sum() / n,
            "one_off_accuracy": rat["almost"].sum() / n,
            "mae": rat["s2"].sum() / n,
            "expected_rating_accuracy": rat["true_pond"].sum() / n,
            "expected_rating_mae": rat["s2pond"].sum() / n,
            "s2": rat["s2"].sum(),
            "s2pond": rat["s2pond"].sum(),
        }

    def compute_likelihood(self, data, theta, eta, pr):
        """Log-likelihood of the standard model, Σ log p(r | u, i).

        Delegates to :meth:`ExpectationMaximization.compute_likelihood`.
        """
        return self.em.compute_likelihood(data, theta, eta, pr)
