import argparse

import numpy as np
from ruamel.yaml import YAML


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mixed membership stochastic block models"
    )
    parser.add_argument(
        "-t",
        "--train",
        dest="train_set",
        type=str,
        help="Train set file name inside 'data' directory.",
        required=False,
    )
    parser.add_argument(
        "-e",
        "--test",
        dest="test_set",
        type=str,
        help="Test set file name inside 'data' directory.",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        type=str,
        help="Data file name inside 'data' directory. You also need to specify "
        "the number of folds for cross-validation you want.",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--n_folds",
        dest="n_folds",
        type=int,
        help="Number of folds for cross-validation. This goes together with data.",
        required=False,
    )
    parser.add_argument(
        "-k",
        "--user_groups",
        dest="K",
        type=int,
        help="Number of user groups",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--item_groups",
        dest="L",
        type=int,
        help="Number of item groups",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        dest="iterations",
        default=200,
        help="How many iterations should I do?",
    )
    parser.add_argument(
        "-s",
        "--sampling",
        type=int,
        dest="sampling",
        default=1,
        help="How many full runs should I do?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=None,
        help="If you'd like, set a random seed.",
    )

    args = parser.parse_args()

    return (
        args.train_set,
        args.test_set,
        args.data,
        args.n_folds,
        args.K,
        args.L,
        args.iterations,
        args.sampling,
        args.seed,
    )


def import_config(local=True):
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.preserve_quotes = True
    yaml.boolean_representation = ["False", "True"]

    if local:
        with open("local_config.yml", "r") as yml_file:
            cfg = yaml.load(yml_file)
    else:
        try:
            with open("../config.yml", "r") as yml_file:
                cfg = yaml.load(yml_file)
        except FileNotFoundError:
            with open("config.yml", "r") as yml_file:
                cfg = yaml.load(yml_file)

    return cfg


def _invert_dict(d):
    return {v: k for k, v in d.items()}


def get_n_per_group(x, n):
    for i in reversed(range(n)):
        try:
            return np.random.choice(x.index, i + 1, replace=False)
        except ValueError:
            pass


def structure_folds(data, folds):
    # How many different items we have?
    n_items = len(set(data.iloc[:, 1]))
    # Check that we haven't asked for too many folds
    assert (
        folds <= n_items
    ), f"Fold number can't be higher than {n_items} since this is the number of different items you have."
    # Return the number of items that we can have in each fold
    return int(n_items / folds)
