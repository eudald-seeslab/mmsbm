from expectation_maximization import (
    normalize_with_d,
    init_random_array,
    normalize_with_self,
    update_coefficients,
    compute_likelihood,
    compute_prod_dist,
)
from data_handler import DataHandler
from utils import get_n_per_group, structure_folds, _invert_dict
