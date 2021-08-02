from src.utils import parse_args
from src.mmsbm import MMSBM


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
        data=pd.read_csv("data/train_5.csv"),
        user_groups=user_groups,
        item_groups=item_groups,
        iterations=iterations,
        sampling=sampling,
        seed=1714,
    )
    return_dict = mmsbm.train()
    results = mmsbm.test(return_dict)
