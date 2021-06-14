# Mixed Membership Stochastic Block Models

This repo follows [this](https://github.com/agodoylo/MMSBMrecommender) 
other work on Mixed Membership Stochastic Block Models to build a recommender 
system [1].

## Usage

### Setup

Install dependencies from requirements.txt:

```
pip install -r requirements.txt
```

### Run

You have three ways of running this: 

#### Normal run

If you know the parameters you want to use, you can do:

```
python run.py --args
```

with the following arguments:
- `-t` or `--train`: train set name (already within the data directory)
- `-e` or `--test`: test set name (already within the data directory)
- `-k` or `--user_groups`: number of user groups to consider.
- `-l` or `--item_groups`: number of item groups to consider.
- `-i` or `--iterations`: number of iterations for each pass (default=200).
- `-s` or `--sampling`: number of passes or runs (default=1).
- `--seed`: if you want, you can give me a seed for pseudorandom initialization.

The script prints out the model goodness of fit parameters.

#### Jupyter notebook

Furthermore, you can use the Runner.ipynb notebook. It calls the same script as
run.py, but it returns the goodness of fit parameters and the predictions. You 
can then use those, for example, to check convergence of the model parameters.

#### Optimizer

If you want to explore and try to find the optimal user and item groups, you 
can do this with the optimizer. To use it, you first need to make a copy of the
`config.yml` file and call it `local_config.yml`. In here you can set both the 
optimizing and training parameters, as well as the number of runs. Make sure 
you name each study differently if you change the parameters, otherwise the 
hyperparameter optimization will go crazy.

After the optimization is complete, you will get the parameters of the best run
and can pass them to run.py.

*Note:* the optimization results will be stored in a self-generated database
called parameters.db. If you stop it and start it again, it will continue where 
it left off.

**Optuna dashboard**

The library for optimization is optuna. They have released a nice component whereby
you can create a dashboard to visualize the progress. To use it, you can do:

```
optuna-dashboard sqlite:///parameters.db
```

### Performance

Each iteration takes a little less than a second in my Intel i7. This means that a
400 iteration runs takes around 6 minutes and a half. The computation 
is vectorized and parallelized over sampling, so, if you choose less sampling
than number of cores, you should have approximately the same performance regardless
of sampling size.

A complete study could be something like 100 hyperparameter optimization runs
of 6 samples of 400 iterations, which will take about 10 hours. 

### Tests

There are a few tests, you can run them with:

```
python -m unittest discover
```

Note: I don't know if they work anymore. 

## TODO

- Fix crazy error bars
- Fix and add tests
- Add visualizations
- Add support for different datasets


# References
[1]: Godoy-Lorite, Antonia, et al. "Accurate and scalable social recommendation 
using mixed-membership stochastic block models." Proceedings of the National 
Academy of Sciences 113.50 (2016): 14207-14212.
