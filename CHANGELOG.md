# Changelog

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