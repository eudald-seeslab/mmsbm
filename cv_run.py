import logging
import os

import numpy as np
import pandas as pd

from lib.utils import parse_args
from lib.mmsbm import MMSBM


TRAIN_NAME = "cv_train.csv"
TEST_NAME = "cv_test.csv"


if __name__ == "__main__":
    # Note: this is overly complicated because I want to reuse the runner from a
    # jupyter notebook
    (
        _,
        _,
        data,
        n_folds,
        user_groups,
        item_groups,
        iterations,
        sampling,
        seed,
    ) = parse_args()

    logger = logging.getLogger("MMSBM")
    logging.basicConfig(level=logging.INFO)

    # There are only 8 or 9 tests; n_folds bigger than that will cause problems
    assert n_folds < 9, (
        "Please make the n_folds smaller that 9 since we" "only have 9 tests."
    )

    path_ = os.path.join("data", data)
    df = pd.read_csv(path_, sep=None, usecols=[0, 1, 2], engine="python", header=None)
    df.columns = ["student", "test", "rating"]

    def get_one(x):
        return np.random.choice(x.index, 1, False)

    accuracies = []
    temp = df
    leftover_indices = df.index
    for n in range(n_folds):

        logger.info(f"Running fold {n + 1} of {n_folds}...")

        # Get the correct indices
        test_indices = [
            a[0] for a in temp.groupby("student", as_index=False).apply(get_one).values
        ]
        # FIXME: for some reason, I get a zero here...
        if 0 in test_indices:
            test_indices.remove(0)

        train_indices = [a for a in df.index if a not in test_indices]
        leftover_indices = [a for a in leftover_indices if a not in test_indices]
        temp = df.iloc[leftover_indices, :]

        # Crate the train and test sets for each fold
        # FIXME: this is objectively sub-optimal, but otherwise I need to make
        #  a ton of changes.
        train = df.iloc[train_indices, :].to_csv(os.path.join("data", TRAIN_NAME), index=False)
        test = df.iloc[test_indices, :].to_csv(os.path.join("data", TEST_NAME), index=False)

        mmsbm = MMSBM(
            train_set=TRAIN_NAME,
            test_set=TEST_NAME,
            user_groups=user_groups,
            item_groups=item_groups,
            iterations=iterations,
            sampling=sampling,
            seed=1714,
            notebook=True,
        )
        return_dict = mmsbm.train()
        s_prs, accuracy, mae, s2, s2pond, rat, lkh, theta, eta = mmsbm.test(
            return_dict
        )
        accuracies.append(accuracy)

    logger.info(f"Ran {n_folds} folds with accuracies {accuracies}.")
    logger.info(f"They have mean {np.mean(accuracies)} and sd {np.std(accuracies)}.")
