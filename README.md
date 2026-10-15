# Mixed Membership Stochastic Block Models

[![PyPI version](https://badge.fury.io/py/mmsbm.svg)](https://badge.fury.io/py/mmsbm)
[![Documentation Status](https://readthedocs.org/projects/mmsbm-docs/badge/?version=latest)](https://mmsbm-docs.readthedocs.io/en/latest/?badge=latest)
[![Python Versions](https://img.shields.io/pypi/pyversions/mmsbm.svg)](https://pypi.org/project/mmsbm/)
[![Tests](https://github.com/eudald-seeslab/mmsbm/actions/workflows/tests.yml/badge.svg)](https://github.com/eudald-seeslab/mmsbm/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/eudald-seeslab/mmsbm/badge.svg?branch=main)](https://coveralls.io/github/eudald-seeslab/mmsbm?branch=main)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Downloads](https://pepy.tech/badge/mmsbm)](https://pepy.tech/project/mmsbm)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15011623.svg)](https://doi.org/10.5281/zenodo.15011623)

A Python implementation of Mixed Membership Stochastic Block Models for recommendation systems, based on the work by Godoy-Lorite et al. (2016). This library provides an efficient, vectorized implementation with multiple computational backends suitable for both research and production environments.

## Features

- **Multiple Backends**: Choose between `numpy` (default), `numba` (JIT-compiled CPU), and `cupy` (GPU-accelerated) for performance tuning.
- Fast, vectorized implementation of MMSBM.
- Support for both simple and cross-validated fitting.
- Parallel processing for multiple sampling runs.
- Comprehensive model statistics and evaluation metrics.
- Optional **degree-corrected** variant (`degree_corrected=True`): the mixed-membership analogue of the degree-corrected stochastic block model, which also models which user-item pairs are observed and lets node degrees be absorbed by the parameters instead of driving the groups. Same EM kernels, same cost per iteration.
- Model selection: `select_groups` chooses the number of user and item groups by cross-validated held-out log-likelihood.
- Compatible with Python 3.7+.

## Installation

The base library can be installed with pip:
```bash
pip install mmsbm
```

For accelerated backends, you can install the optional dependencies:

**Numba (JIT Compilation on CPU):**
```bash
pip install mmsbm[numba]
```

**CuPy (NVIDIA GPU Acceleration):**
Make sure you have a compatible NVIDIA driver and CUDA toolkit installed. Then install with:
```bash
pip install mmsbm[cupy]
```
> **Note**: The CuPy backend requires **Python 3.12 or lower**. As of December 2025, CuPy does not provide pre-built wheels for Python 3.13+.

You can also install all optional dependencies with:
```bash
pip install mmsbm[numba,cupy]
```

## Performance & Backends

This library uses a backend system to perform the core computations of the Expectation-Maximization algorithm. You can specify the backend when you initialize the model, giving you control over the performance characteristics.

```python
from mmsbm import MMSBM

# Use the default, pure NumPy backend
model_numpy = MMSBM(user_groups=2, item_groups=4, backend='numpy')

# Use the Numba backend for JIT-compiled CPU acceleration
model_numba = MMSBM(user_groups=2, item_groups=4, backend='numba')

# Use the CuPy backend for GPU acceleration
model_cupy = MMSBM(user_groups=2, item_groups=4, backend='cupy')
```

- **`numpy` (Default)**: A highly optimized, pure NumPy implementation. It is universally compatible and requires no extra dependencies beyond NumPy itself.
- **`numba`**: Uses the Numba library to just-in-time (JIT) compile the core computational loops. This can provide a significant speedup on the CPU, especially for large datasets. It is recommended for users who want better performance without a dedicated GPU. Note that there is some issue with the parallelization of samples which makes numba slower for smaller datasets.
- **`cupy`**: Offloads computations to a compatible NVIDIA GPU using the CuPy library. This provides the best performance but requires a CUDA-enabled GPU and the appropriate drivers. Note that there is some overhead for transferring data to and from the GPU, so it's most effective on larger models where the computation time outweighs the data transfer time.

> **Note**: There are some issues with numba parallelization and cupy data transfer to the GPU which don't guarantee they will be superior to Numpy. Please try different backends to find the best one for your data and your system.


## Usage

### Data Format

The input data should be a pandas DataFrame with exactly 3 columns calles _users_, _items_, and _ratings_. For example:

```python
import pandas as pd
from random import choice

train = pd.DataFrame(
    {
    "users": [f"user{choice(list(range(5)))}" for _ in range(100)],
    "items": [f"item{choice(list(range(10)))}" for _ in range(100)],
    "ratings": [choice(list(range(1, 6))) for _ in range(100)]
    }
)

test = pd.DataFrame(
    {
    "users": [f"user{choice(list(range(5)))}" for _ in range(50)],
    "items": [f"item{choice(list(range(10)))}" for _ in range(50)],
    "ratings": [choice(list(range(1, 6))) for _ in range(50)]
    }
)

```

### Model Configuration

```python

from mmsbm import MMSBM

# Initialize the MMSBM class:
model = MMSBM(
    user_groups=2,      # Number of user groups
    item_groups=4,      # Number of item groups
    backend='numba',    # Specify the computational backend (numpy, numba, or cupy)
    iterations=500,     # Number of EM iterations
    sampling=5,         # Number of parallel EM runs (different random initializations); the best run is kept
    seed=1,             # Random seed for reproducibility
    debug=False,        # Enable debug logging
    degree_corrected=False  # Fit the degree-corrected variant (see below)
)
```

> **Note on `sampling`**  
> Setting `sampling` to a value greater than 1 makes the library launch that many independent EM optimizations in parallel, each starting from a different random initialization. The predicted probabilities average all runs; the run with the highest training log-likelihood is the one exposed as `theta`, `eta`, `pr` and `likelihood`. This increases the chances of finding a better (global) solution at the cost of extra computation time.


### Training Methods

The library offers two complementary ways to train a model:

* **Simple Fit** – runs the EM algorithm once on the full training set. This is the fastest option and is appropriate when you already have a train-test split (or when you do not need a validation step).
* **Cross-Validation Fit** – automatically splits the input data into *k* folds (default 5), trains a separate model on each *(k-1)* subset, and evaluates it on the held-out fold. Each user's ratings are dealt round-robin across the folds, so every user with at least two ratings stays in training and the folds have the same size; held-out ratings whose user or item is absent from the training folds are left out. The routine returns the accuracy of every fold so you can inspect the variability and pick hyper-parameters more reliably. It is slower because it performs the fit *k* times but provides an unbiased estimate of generalisation performance.

#### Simple Fit

```python
model.fit(train)
```

#### Cross-Validation Fit

```python
accuracies = model.cv_fit(train, folds=5)
print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
```

### Making Predictions

```python
predictions = model.predict(test)
```

### Model Evaluation

> **Note**: you need to predict before running model.score().

```python
results = model.score()

# Access various metrics
accuracy = results['stats']['accuracy']
mae = results['stats']['mae']

# Access model parameters
theta = results['objects']['theta']  # User group memberships
eta = results['objects']['eta']      # Item group memberships
pr = results['objects']['pr']        # Rating probabilities
```

Two point predictions are evaluated for every held-out rating: the *most probable* rating, `argmax_r p(r | u, i)`, and the *expected* rating, `Σ_r r p(r | u, i)`. Errors are measured in rating units when the rating labels are numbers (half stars or ratings from 1 to 10 are handled correctly) and in positions of the sorted list of distinct ratings otherwise.

| `stats` key | Meaning |
|---|---|
| `accuracy` | fraction of ratings equal to the most probable rating |
| `one_off_accuracy` | fraction of ratings within one unit of the most probable rating |
| `mae` | mean absolute error of the most probable rating |
| `expected_rating_accuracy` | fraction of ratings equal to the rating level nearest to the expected rating |
| `expected_rating_mae` | mean absolute error of the expected rating |
| `s2`, `s2pond` | sums of absolute errors of the most probable and of the expected rating (`mae` and `expected_rating_mae` times the number of predictions; kept for compatibility) |
| `likelihood` | log-likelihood of the training data, `Σ log p(r_ui | u, i)`, of the run exposed in `objects` (the Poisson log-likelihood for the degree-corrected model) |

## Degree-corrected model

The standard MMSBM only models the rating *given* that a user has rated an item: the probability of a rating is conditional on the (user, item) pair being observed, so the model never sees which items each user rates, nor how many. Degree heterogeneity therefore cannot affect the groups, but neither can the structure of the bipartite graph of who-rates-what.

`degree_corrected=True` fits the mixed-membership analogue of the degree-corrected stochastic block model [2, 3, 4]: every (user, item, rating) cell is a Poisson count with rate

    λ_uir = Σ_kℓ θ_uk η_iℓ p_kℓ(r)

and the memberships θ and η are left unnormalised, so that their row sums absorb the degrees. The model then explains both which pairs are rated and how they are rated, while popularity is captured by per-node scales instead of by the groups.

```python
model = MMSBM(user_groups=3, item_groups=4, iterations=500, sampling=5, degree_corrected=True)
model.fit(train)
predictions = model.predict(test)   # p(r | u, i), exactly as in the standard model
results = model.score()

# Probability that a (user, item) pair is rated at all (degree-corrected model only)
pairs = test[["users", "items"]]
link_probabilities = model.predict_links(pairs)
```

What changes with the flag on:

- Each row of `theta` and `eta` sums to the number of ratings of that user or item (its *degree*); divide a row by its sum to recover the membership distribution. `pr[k, ℓ, r]` holds the Poisson rate per pair of half-edges, `e_kℓ(r) / (e_k e_ℓ)`, where `e_kℓ(r)` is the expected number of ratings `r` between user group `k` and item group `ℓ` and `e_k`, `e_ℓ` are the numbers of ratings attached to each group; `pr[k, ℓ, r] / Σ_r' pr[k, ℓ, r']` is the rating distribution of the pair, as in the standard model.
- `results[i]["likelihood"]` and `stats["likelihood"]` hold the Poisson log-likelihood.
- `predict` returns p(r | u, i) exactly as before, so accuracies and errors are comparable across the two models, and `predict_links` becomes available.
- The EM kernels (numpy, numba, cupy) are shared with the standard model; only the M-step normalisation, the likelihood and the prediction step differ, so the cost per iteration is the same.
- The rating probabilities are initialised with a random 10:1 affinity pattern between user and item groups, because a uniform affinity is a stationary point of the Poisson likelihood from which the EM cannot escape. As with the standard model, use `sampling > 1` to guard against local optima.

The degree-corrected model fits links and ratings jointly: on data where the groups are only visible in the link pattern it recovers them and the standard model cannot, but it is not guaranteed to predict ratings better, because part of its parameters are spent on the links. Compare the two with `held_out_log_likelihood` or `select_groups`.

With the flag off (the default) the code path, the random numbers and the results are identical to previous versions.

## Choosing the number of groups

`select_groups` runs a cross-validated grid search over the number of user and item groups and scores every combination by the mean held-out log-likelihood of the ratings, `log p(r_ui | u, i)`, which is defined identically for the standard and the degree-corrected model. Each user's ratings are spread across folds, so every user with at least two ratings stays in training.

```python
model = MMSBM(user_groups=2, item_groups=2, iterations=300, sampling=3, seed=1, degree_corrected=True)
grid = model.select_groups(train, user_groups=[2, 3, 4, 5], item_groups=[2, 4, 6], folds=5)
print(grid.sort_values("log_likelihood", ascending=False))

best = MMSBM(*model.selected_groups, iterations=300, sampling=3, degree_corrected=True)
best.fit(train)
```

The held-out log-likelihood is used instead of a penalised training likelihood (AIC, BIC) because the number of membership parameters grows with the number of users and items, outside the regime where such penalties are justified. Because the fits are maximum likelihood, the predicted probabilities are smoothed towards the uniform distribution (`smoothing=0.05` by default) before taking logs; otherwise a handful of held-out ratings predicted with near-zero probability dominate the mean and favour too few groups. `criterion="accuracy"` selects by held-out accuracy instead, and the grid reports both. The grid costs `len(user_groups) × len(item_groups) × folds` fits, so start with a coarse grid and few iterations. After any `predict`, `model.held_out_log_likelihood()` gives the same criterion for a single fit.

## Running Tests

To run tests do the following:

```
pytest
```

## Contributing

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## TODO

- Progress bars are not working for jupyter notebooks.
- There is a persistent (albeit harmless) warning when using the cupy backend.
- Numba and cupy backends show unexpected behaviour on the "sampling" parallelization.
- Improve the treatment of the prediction / score steps and the results object
- Add sampling as an extra axis in the EM objects for more efficiency


# References
[1]: Godoy-Lorite, Antonia, et al. "Accurate and scalable social recommendation using mixed-membership stochastic block models." Proceedings of the National Academy of Sciences 113.50 (2016): 14207-14212.

[2]: Ball, Brian, Brian Karrer, and M. E. J. Newman. "Efficient and principled method for detecting communities in networks." Physical Review E 84.3 (2011): 036103.

[3]: Karrer, Brian, and M. E. J. Newman. "Stochastic blockmodels and community structure in networks." Physical Review E 83.1 (2011): 016107.

[4]: Peixoto, Tiago P. "Bayesian stochastic blockmodeling." In Advances in Network Clustering and Blockmodeling, Wiley (2019). arXiv:1705.10225.
