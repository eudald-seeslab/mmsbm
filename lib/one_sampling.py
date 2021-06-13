import numpy as np
from tqdm import tqdm

from lib.funcs import normalize_with_d, init_random_array, normalize_with_self, update_coefs, compute_likelihood, \
    compute_prod_dist


def run_one_sampling(d0, d1, p, m, r, user_groups, item_groups, iterations, train, test, ratings, seed, i, return_dict):
    rng = np.random.default_rng(seed)

    # Generate random (but normalized) inits
    theta = normalize_with_d(
        init_random_array((p + 1, user_groups), rng), d0
    )
    eta = normalize_with_d(
        init_random_array((m + 1, item_groups), rng), d1
    )
    pr = normalize_with_self(
        init_random_array((user_groups, item_groups, r), rng)
    )

    # Do the work
    # We store the prs to check convergence
    prs = []
    for _ in tqdm(range(iterations)):
        # This is the crux of the script; please see funcs.py
        n_theta, n_eta, npr = update_coefs(data=train, ratings=ratings, theta=theta, eta=eta, pr=pr)

        # Update with normalization
        theta = normalize_with_d(n_theta, d0)
        eta = normalize_with_d(n_eta, d1)
        pr = normalize_with_self(npr)

        # This can be removed when not debugging
        prs.append(pr)

    likelihood = compute_likelihood(train, ratings, theta, eta, pr)
    rat = compute_prod_dist(test, theta, eta, pr)

    return_dict[i] = {"likelihood": likelihood, "rat": rat, "prs": prs}

    return None
