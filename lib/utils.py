import argparse

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
        required=True,
    )
    parser.add_argument(
        "-e",
        "--test",
        dest="test_set",
        type=str,
        help="Test set file name inside 'data' directory.",
        required=True,
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


