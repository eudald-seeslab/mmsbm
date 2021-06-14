from lib.utils import parse_args
from lib.mmsbm import mmsbm


if __name__ == "__main__":
    # Note: this is overly complicated because I want to reuse the runner from a
    # jupyter notebook
    (
        train_set,
        test_set,
        user_groups,
        item_groups,
        iterations,
        sampling,
        seed,
    ) = parse_args()
    mmsbm(train_set, test_set, user_groups, item_groups, iterations, sampling, seed)
