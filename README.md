# Under construction

This repo loosely follows [this](https://github.com/agodoylo/MMSBMrecommender) 
other work on Mixed Membership Stochastic Block Models to build a recommender 
system.

## Usage

### Setup

Install dependencies from requirements.txt:

```
pip install -r requirements.txt
```

### Run

To run do:

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

### Tests

There are a few tests, you can run them with:

```
python -m unittest discover
```


## TODO
Many things...