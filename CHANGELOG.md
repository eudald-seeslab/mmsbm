# Changelog

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