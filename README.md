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
    debug=False         # Enable debug logging
)
```

> **Note on `sampling`**  
> Setting `sampling` to a value greater than 1 makes the library launch that many independent EM optimizations in parallel, each starting from a different random initialization. Once all runs finish, the one with the highest log-likelihood is selected. This increases the chances of finding a better (global) solution at the cost of extra computation time.


### Training Methods

The library offers two complementary ways to train a model:

* **Simple Fit** – runs the EM algorithm once on the full training set. This is the fastest option and is appropriate when you already have a train-test split (or when you do not need a validation step).
* **Cross-Validation Fit** – automatically splits the input data into *k* folds (default 5), trains a separate model on each *(k-1)* subset, and evaluates it on the held-out fold. The routine returns the accuracy of every fold so you can inspect the variability and pick hyper-parameters more reliably. It is slower because it performs the fit *k* times but provides an unbiased estimate of generalisation performance.

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
