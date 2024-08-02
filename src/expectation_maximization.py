# These are the functions that carry out the Expectation-Maximization procedure

import numpy as np


def compute_omegas(data, theta, eta, pr):
    user_indices = data[:, 0]
    item_indices = data[:, 1]
    rating_indices = data[:, 2]

    # Shape: (input_length, num_user_groups, 1)
    theta_expanded = theta[user_indices][:, :, np.newaxis]
    # Shape: (input_length, 1, num_item_groups)
    eta_expanded = eta[item_indices][:, np.newaxis, :]
    # Shape: (num_user_groups, num_item_groups, input_length)
    pr_selected = pr[:, :, rating_indices]

    # Adjust pr_selected to match dimensions for broadcasting
    # Shape: (input_length, num_user_groups, num_item_groups)
    pr_expanded = np.moveaxis(pr_selected, -1, 0)

    # Returned shape: (input_length, num_user_groups, num_item_groups)
    return theta_expanded * (eta_expanded * pr_expanded)


def update_coefficients(data, ratings, theta, eta, pr):

    # Shape: (input_length, num_user_groups, num_item_groups)
    omegas = compute_omegas(data, theta, eta, pr)
    # Shape: (input_length)
    sum_omega = omegas.sum(axis=-1).sum(axis=-1)
    # Shape: (input_length, num_user_groups, num_item_groups)
    increments = omegas / sum_omega[:, np.newaxis, np.newaxis]

    # You may want to vectorize this, but, for some reason, it ends up being slower
    n_theta = np.array(
        [
            increments[np.where(data[:, 0] == a)].sum(-1).sum(0)
            for a in range(theta.shape[0])
        ]
    )
    n_eta = np.array(
        [
            increments[np.where(data[:, 1] == a)].sum(0).sum(0)
            for a in range(eta.shape[0])
        ]
    )
    n_pr = np.swapaxes(
        np.swapaxes(
            np.array([increments[np.where(data[:, 2] == a)].sum(0) for a in ratings]),
            0,
            1,
        ),
        1,
        2,
    )

    return n_theta, n_eta, n_pr


def normalize_with_d(df, d):
    return df / [np.repeat(max(len(a), 1), df.shape[1]) for a in list(d.values())]


def normalize_with_self(df):
    # Note: only valid for 3d arrays
    temp = df.reshape((df.shape[0] * df.shape[1], df.shape[2]))
    return (
        temp / (np.where(temp.sum(axis=1) == 0, 1, temp.sum(axis=1)))[:, np.newaxis]
    ).reshape(df.shape)


def compute_likelihood(data, theta, eta, pr):
    omegas = compute_omegas(data, theta, eta, pr)
    return sum([a * np.log(b) / b for (a, b) in zip(omegas, omegas.sum(-1).sum(-1))])


def prod_dist(x, theta, eta, pr):
    return (
        (theta[x[0]][:, np.newaxis, np.newaxis] * (eta[x[1], :][:, np.newaxis] * pr))
        .sum(axis=0)
        .sum(axis=0)
    )


def compute_prod_dist(data, theta, eta, pr):
    return np.array([prod_dist(a, theta, eta, pr) for a in data])
