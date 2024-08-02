# Mixed Membership Stochastic Block Models

[![Build Status](https://travis-ci.com/eudald-seeslab/mmsbm.svg?token=FgqRjRbiBxssKd9AcHMK&branch=main)](https://travis-ci.com/eudald-seeslab/mmsbm)

This library converts [this](https://github.com/agodoylo/MMSBMrecommender) 
 work on Mixed Membership Stochastic Block Models to build a recommender 
system [1] into a library to be used with more generic data.

## Installation

```
pip install mmsbm
```

## Usage

### Input data

You'll need a pandas dataframe with exactly 3 columns: users, items and ratings, e.g.:

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

### Setup

```python

from mmsbm import MMSBM

# Initialize the MMSBM class:
mmsbm = MMSBM(
    user_groups=2,
    item_groups=4,
    iterations=500,
    sampling=5,
    seed=1,
)
```

### Fit models

In here you have two options, a simple fit where we run "sampling" times the fitting algorithm and return the results
for all runs, you are then in charge of choosing the best one. 

```python
mmsbm.fit(train)
```

The other option is the cv_fit (cross-validated fit) function, whereby we split the input data in "folds" number of folds
and run the fitting in each one and test on the excluded fold. We then return all the 
samplings of the best performing model. The function returns a list of the accuracies for 
each fold so that you can get confidence intervals on them.

```python
accuracies = mmsbm.cv_fit(train, folds=5)
```

### Prediction

Once the model is fitted, we can predict on test data. The function predict returns
the prediction matrix (the probability of each user to belong to each group) as a numpy array.

```python
pred_matrix = mmsbm.predict(test)
```

### Score

Finally, you can get statistics about the goodness of fit and other parameters of the model, 
as well as the computed objects: the theta matrix, the eta matrix and the probability distributions.

The function score returns a dictionary with two sub-dictionaries, one for statistics about the model (called "stats") and 
the other one with the computed objects (called "objects").

```python
results = mmsbm.score()
```

## Performance

Each iteration takes a little about half a second in an Intel i7. This means that a
500 iteration runs takes around 4 minutes. The computation is vectorized, so, as 
long as you don't go crazy with the number of observations, the time should be 
approximately the same regardless of training set size. It is also parallelized 
over sampling, so, as long as you choose less sampling than number of cores, 
you should have approximately the same performance  regardless of training set 
size and sampling number.

## Tests

To run tests do the following:

```
python -m pytest tests/*
```


## TODO

- Progress bars are not working for jupyter notebooks.
- Include user_groups and item_groups optimization procedure.
- The cv_fit test is not working on travis.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

# References
[1]: Godoy-Lorite, Antonia, et al. "Accurate and scalable social recommendation 
using mixed-membership stochastic block models." Proceedings of the National 
Academy of Sciences 113.50 (2016): 14207-14212.
