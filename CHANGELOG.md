# Changelog

## [2.0.0] - 2026-10-15

`mmsbm` is now a package. Nothing that worked with 1.4 stops working: everything that moved or changed keeps a compatibility path for the whole 2.x series and emits a `FutureWarning` that names the replacement (see the deprecation policy below). The only visible changes for a program that uses the documented API are the exception classes, which are still `AssertionError` subclasses, and the log file, which is no longer written unless asked for.

### Changed
- Package layout: the modules used to be installed at the top level of `site-packages` (`import helpers`, `import logger`, ...), where they could shadow, or be shadowed by, any other module of those names. They are now `mmsbm.model` (the `MMSBM` class), `mmsbm.data_handler`, `mmsbm.expectation_maximization`, `mmsbm.helpers`, `mmsbm.backend`, `mmsbm.logger`, `mmsbm.kernels_numpy`, `mmsbm.kernels_numba`, `mmsbm.kernels_cupy` and `mmsbm.exceptions`. `from mmsbm import MMSBM` is unchanged; `mmsbm` also exports `DataHandler`, `ExpectationMaximization`, `MMSBMError`, `NotFittedError` and `__version__`, and `MMSBM.__module__` is `"mmsbm"`, so pickles keep the 1.x path `mmsbm.MMSBM`.
- Errors: invalid input and calls in the wrong order raise `mmsbm.MMSBMError`, a `ValueError`, or its subclass `mmsbm.NotFittedError` (also an `AttributeError`, as in scikit-learn) instead of `AssertionError`: `predict` before `fit`, `score` or `held_out_log_likelihood` before `predict`, `predict_links` on the standard model, `folds`, `smoothing` or `criterion` out of range, missing values, and scoring data without ratings. The `assert` statements they replace were removed by `python -O`, which silently disabled every check.
- `MMSBM(log_file=None)` is the default: the log goes to the console unless a file is asked for. 1.x wrote `mmsbm.log` in the working directory of every program that used the package. `logger.setup_logger` has the same new default.
- The API reference of the documentation is generated from the package (`docs/api/modules.rst`).

### Fixed
- Importing the package set `pandas.options.mode.chained_assignment = None`, which silenced pandas' `SettingWithCopyWarning` in the whole program that used it. The option is no longer touched; the encoding of the data no longer writes into frames, so it raises no such warning itself with pandas 2 either.
- Missing values in the data were never detected: the check ran after the values had been converted to strings, so a `NaN` or `None` became a user, item or rating called `"nan"` or `"None"`. The check now runs first and raises `MMSBMError`.

### Deprecated, removed in 3.0 (not before October 2027)
- The top-level modules `helpers`, `logger`, `backend`, `data_handler`, `expectation_maximization`, `kernels_numpy`, `kernels_numba` and `kernels_cupy`: they are aliases of `mmsbm.<name>` that expose the same objects and warn once per process when imported. Pickles of 1.x models, which refer to `data_handler.DataHandler` and `expectation_maximization.ExpectationMaximization`, load through them.
- `mmsbm.assign_folds`, `mmsbm.drop_unseen` and `mmsbm.setup_logger`, which the module `mmsbm` of 1.x exposed because it imported them; use `mmsbm.helpers` and `mmsbm.logger`.
- The `AssertionError` base of `MMSBMError`, which keeps `except AssertionError` working. No warning can be emitted for this one.
- The `rats` argument of `choose_best_run`, ignored since 1.2.0 (warns when passed).
- `MMSBM.d0` and `MMSBM.d1`, the dictionaries of neighbours, which nothing in the package has used since 1.1.1 (warn when read).

### Deprecation policy
A deprecated feature emits a `FutureWarning` that names its replacement, keeps working for the whole major series in which it was deprecated, and is removed in the next major version, which is published at least twelve months after the deprecation. `FutureWarning` rather than `DeprecationWarning` because Python hides the latter outside `__main__` and test runners, and the package is mostly used from scripts and notebooks. `python -W error::FutureWarning` turns them into errors.

## [1.4.0] - 2026-10-15

### Added
- `MMSBM(..., user_col="users", item_col="items", rating_col="ratings")`: the names of the columns of the data that hold the users, the items and the ratings. `fit`, `predict`, `predict_links`, `cv_fit` and `select_groups` look the columns up by these names, in any order, and ignore any other column (`DataHandler.select_columns`).
- `predict` and `predict_links` accept data without a rating column, since the predictions do not need it; `score` and `held_out_log_likelihood` then raise `AssertionError` saying so.
- The data can also be a list of `(user, item, rating)` tuples or a 2-D array, read by position.

### Changed
- Column handling mixed position and name: `fit` read the first three columns whatever their names, while the "not in the training set" filter of `predict` (and so of `cv_fit` and `select_groups`) looked for the names `users`, `items` and `ratings` and raised `KeyError` otherwise. With the right names in another order, or an extra column before them, `fit` trained silently on the wrong columns and `predict` then dropped every row. Now every entry point selects the columns by name. A DataFrame with none of the names and exactly as many columns as needed (three, or two for `predict` and `predict_links`) is still read by position, now with a warning that says which column is taken as what. A DataFrame with more columns and none of the names, which was read by position silently, raises `ValueError`.

### Removed
- `DataHandler._get_data`, an unused CSV reader.

## [1.3.0] - 2026-10-15

### Added
- `MMSBM(..., chunk_size=None)`: the numpy and cupy kernels process long data in blocks whose partial numerators are summed, so the memory of an iteration is bounded whatever the number of ratings. By default a block holds as many rows as keep the responsibility tensor (rows × K × L, float64) under 256 MiB (`ExpectationMaximization.CHUNK_BUDGET`); data shorter than one block is processed in a single pass, exactly as before. The likelihoods and the prediction step are blocked too.
- `MMSBM(..., log_file="mmsbm.log")`: `None` disables the log file (console only). The default is unchanged.
- `ExpectationMaximization.to_device` / `to_host` / `chunk_rows`.

### Changed
- Numba backend: the M-step runs in parallel over chunks of rows with per-thread partial accumulators (bounded by `kernels_numba.ACCUMULATOR_BUDGET`) and never materialises the responsibility tensor; `prod_dist` is a parallel kernel instead of numpy's einsum; all kernels are compiled with `cache=True`, so worker processes reuse the compiled code; with `sampling > 1` each run uses `NUMBA_NUM_THREADS // sampling` threads instead of oversubscribing the cores.
- CuPy backend: the kernels accept CuPy arrays and then return CuPy arrays, and the EM loop keeps the data, the parameters and the normalisation factors on the GPU for the whole run (one transfer in, one out, instead of eight per iteration). `cupy.add.at` replaces the deprecated `cupyx.scatter_add` when available.
- The numba and cupy kernels floor the responsibility sums with `max(s, eps)` like the numpy kernel (they used `s + eps`), so the three backends compute the same quantities. They agree to about 1e-12 relative, not bit for bit: the order of the floating-point additions differs between the serial, parallel, atomic and blocked paths. With the numpy backend and data shorter than one block the results are still identical to 1.0.7 (regression test).
- `m_step_degree_corrected` and `normalize_rows` are written with `np.where` so that they run unchanged on CuPy arrays; same values.

## [1.2.1] - 2026-10-15

### Fixed
- `MMSBM` docstrings: `debug` no longer claims to set 10 iterations and a single run (it prints the backend and logs the training log-likelihood every 50 iterations); `seed` defaults to `None`, not 1714; `results` is a list with one dictionary per run, not the output of `score`. The `backend` parameter documents `cupy`.
- `mmsbm.__version__` exists and is the single source of the version for `setup.py` and the documentation (`CITATION.cff` is still updated by hand).
- `python_requires`, the README and the installation guide claimed Python 3.7+. The package is tested on Python 3.10 to 3.13 and now says so; the CI matrix adds 3.13. No code changed.
- Coverage measured only `mmsbm.py` (`--cov=mmsbm`); every module of the package is now measured, locally and in CI.

### Removed
- `helpers.structure_folds` and `helpers.get_n_per_group`, the pieces of the pre-1.2.0 `cv_fit` split, which nothing used any more.

## [1.2.0] - 2026-10-15

### Changed
- `cv_fit` assigns ratings to folds with `helpers.assign_folds` (each user's ratings dealt round-robin across the folds), as `select_groups` already did, and leaves out held-out ratings whose user or item is absent from training. The previous split took from every user a fixed number of ratings equal to the number of distinct items divided by `folds`: on sparse data (users with fewer ratings than that) the first fold took all the ratings and the remaining folds were empty, the test parts overlapped with later training parts, and the row labelled `0` never entered a test set. The fold accuracies returned for a given seed therefore differ from earlier versions. `folds` must now be between 2 and the number of ratings. `helpers.structure_folds` and `helpers.get_n_per_group` are kept but no longer used.
- `score()`: `mae` is now the mean absolute error of the most probable rating. Before it was one minus the accuracy of the rounded expected rating, which is now reported as `expected_rating_accuracy`. New `expected_rating_mae`, the mean absolute error of the expected rating (the former `s2pond` divided by the number of predictions). `s2` and `s2pond`, documented as squared errors but actually sums of absolute errors, are unchanged and kept for compatibility. Errors are measured in rating units when all rating labels are numeric (so half stars, or ratings from 1 to 10, whose labels sort as `"1", "10", "2", ...`, are handled correctly) and in positions of the sorted list of distinct ratings otherwise, which is what earlier versions always used.
- `likelihood` (in `results[i]`, `model.likelihood` and `stats`) is now the log-likelihood of the training data, Σ log p(r | u, i), for the standard model (`ExpectationMaximization.compute_likelihood`). The previous value was Σ ω log(ω / Σω), the negative weighted entropy of the responsibilities, which is not a log-likelihood and could not be used to compare runs. The degree-corrected model already reported its Poisson log-likelihood.
- With `sampling > 1`, the run exposed as `theta`, `eta`, `pr` and `likelihood` is the one with the highest training log-likelihood, as the documentation stated; it used to be the one with the highest accuracy on the data passed to `predict`, which selected on the test set. The prediction matrix is unchanged: it averages all runs. `choose_best_run` ignores its argument, which is kept for compatibility.
- The EM itself is untouched: with `degree_corrected=False` the fitted parameters and the predictions remain identical to 1.0.7 (regression test).

## [1.1.1] - 2026-10-15

### Changed
- Preparing the training objects is now linear in the number of ratings (degrees via `np.bincount`). It scaled with ratings × (users + items) and, on large data, took longer than dozens of EM iterations. The precomputed index lists and the ω buffer, which no kernel used, are gone; `d0` and `d1` are built on first access.
- Mapping results back to the original user, item and rating labels inverts each dictionary once instead of once per element (quadratic in the number of users before).
- With `sampling=1` the EM runs in the calling process instead of a pool with a single worker.
- None of this changes any result: with `degree_corrected=False` the random numbers and the outputs are identical to 1.0.7 (checked bitwise for the numpy and numba backends, and pinned by the regression test) and the cost per iteration is unchanged.


## [1.1.0] - 2026-10-15

### Added
- Degree-corrected variant of the model, `MMSBM(..., degree_corrected=True)`: the mixed-membership analogue of the degree-corrected stochastic block model (Ball, Karrer & Newman 2011; Peixoto 2019), fitted with the same EM kernels as the standard model. Memberships absorb the node degrees and the model also explains which user-item pairs are observed. Only the M-step normalisation (`ExpectationMaximization.m_step_degree_corrected`), the likelihood (`poisson_log_likelihood`), the initialisation (`initial_affinity`) and the prediction step differ.
- `MMSBM.predict_links` (degree-corrected model only): probability that a (user, item) pair is rated.
- `MMSBM.held_out_log_likelihood`: mean log p(r | u, i) over the data passed to `predict`, for both models.
- `MMSBM.select_groups`: cross-validated grid search over the number of user and item groups by held-out log-likelihood, with `helpers.assign_folds` and `helpers.drop_unseen`.
- `DataHandler.format_pairs` to encode (user, item) pairs without ratings.
- `benchmark_mmsbm.py --degree_corrected`.
- Regression test pinning the standard model to the 1.0.7 reference output (`tests/data/regression_numpy_v1_0_7.npz`), plus tests of the degree-corrected model on data with planted link structure.
- Documentation of the degree-corrected model and of model selection (README, theory and quick start pages).

## [1.0.7] - 2026-04-25

### Fixed
- Data handling: `_to_object_str` now assigns with explicit `dtype=object` so pandas 2.3+ / 3.x does not keep `StringDtype` after the per-cell `str` step, which would still break integer list assignment in `parse_train_data` / `parse_test_data`.

## [1.0.6] - 2026-04-24

### Fixed
- Data handling: `format_train_data` / `format_test_data` no longer use `DataFrame.astype(str)`, which on pandas 2+ can produce `StringDtype` columns that reject list/integer reassignments in `parse_train_data` / `parse_test_data`. Values are now normalized to plain Python `str` in object-dtype columns for compatibility with pandas 1.3+, 2.x, and 3.x.

## [1.0.2] - 2025-10-20

### Fixed
- Data handling: replaced invalid pandas `.iloc` usages with `.loc` in `DataHandler._check_test_in_train`, fixing ValueError during tests.
- All tests passing locally (22 passed, 4 skipped).

## [1.0.0] - 2025-07-01

### Added
- Implemented CuPy backend for GPU acceleration
- Added benchmark script to compare performance across backends (numpy, numba, cupy)

### Changed
- Better overall organization of kernels, including previous numpa and numba

### Fixed
- Nothing


## [0.4.0] - 2025-06-30

### Added
- Implemented numba backend

### Changed
- Major efficiency improvements in numpy

### Fixed
- Nothing

## [0.3.3] - Previous version
- ... 