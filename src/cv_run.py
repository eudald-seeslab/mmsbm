import logging
import os

import numpy as np
import pandas as pd

from src.utils import parse_args, get_one_per_group
from src.mmsbm import MMSBM


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

    path_ = os.path.join("../data", data)
    df = pd.read_csv(path_, sep=None, usecols=[0, 1, 2], engine="python", header=None)
    df.columns = ["student", "test", "rating"]

    accuracies = []
    temp = df
    leftover_indices = df.index
    for n in range(n_folds):

        logger.info(f"Running fold {n + 1} of {n_folds}...")

        # Get the correct indices
        test_indices = [
            a[0]
            for a in temp.groupby(temp.columns[0], as_index=False)
            .apply(get_one_per_group)
            .values
            if a[0] != 0
        ]

        train_indices = [a for a in df.index if a not in test_indices]
        leftover_indices = [a for a in leftover_indices if a not in test_indices]
        temp = df.iloc[leftover_indices, :]

        # Crate the train and test sets for each fold.
        train = df.iloc[train_indices, :]
        test = df.iloc[test_indices, :]

        mmsbm = MMSBM(
            user_groups=user_groups,
            item_groups=item_groups,
            iterations=iterations,
            sampling=sampling,
            seed=1714,
        )
        mmsbm.fit(train)
        pred_matrix = mmsbm.predict(test)
        results = mmsbm.score(pred_matrix)

        accuracies.append(results["stats"]["accuracy"])

    logger.info(f"Ran {n_folds} folds with accuracies {accuracies}.")
    logger.info(f"They have mean {np.mean(accuracies)} and sd {np.std(accuracies)}.")
