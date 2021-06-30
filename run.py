from lib.utils import parse_args
from lib.mmsbm import MMSBM


if __name__ == "__main__":
    # Note: this is overly complicated because I want to reuse the runner from a
    # jupyter notebook
    (
        train_set,
        test_set,
        _,
        _,
        user_groups,
        item_groups,
        iterations,
        sampling,
        seed,
    ) = parse_args()
    mmsbm = MMSBM(
        train_set=train_set,
        test_set=test_set,
        user_groups=user_groups,
        item_groups=item_groups,
        iterations=iterations,
        sampling=sampling,
        seed=1714,
        notebook=True
    )
    return_dict = mmsbm.process()
    s_prs, accuracy, mae, s2, s2pond, rat, lkh, theta, eta = mmsbm.postprocess(return_dict)

