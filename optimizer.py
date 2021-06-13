import optuna

from lib.mmsbm import mmsbm


def optimizer(trial):
    # Constants
    train = "u1.base"
    test = "u1.test"
    sampling = 5
    # Number of groups of users
    k = trial.suggest_int("k", 1, 30)
    # Number of groups of items
    l = trial.suggest_int("l", 1, 30)
    # Iterations
    # To plateau the coefficients the minimum is 600
    iterations = 10

    return mmsbm(
        train_set=train,
        test_set=test,
        user_groups=k,
        item_groups=l,
        iterations=iterations,
        sampling=sampling,
        seed=1714
        )


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="mmsbm",
        storage="sqlite:///parameters.db",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(optimizer, n_trials=40)
