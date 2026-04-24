# Changelog

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