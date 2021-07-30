# These are the functions that carry out the Expectation-Maximization procedure

import numpy as np
import pandas as pd


def init_random_array(shape, rng):
    return rng.random(shape)


def compute_omega(x, theta, eta, pr, ratings):
    return theta[x[0]][:, np.newaxis] * (eta[x[1], :] * pr[:, :, ratings[x[2]]])


def update_coefs(data, ratings, theta, eta, pr):

    omegas = np.array([compute_omega(a, theta, eta, pr, ratings) for a in data])
    sum_omega = omegas.sum(axis=-1).sum(axis=-1)
    increments = np.array([a / b for (a, b) in zip(omegas, sum_omega)])

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


def compute_likelihood(data, ratings, theta, eta, pr):
    omegas = np.array([compute_omega(a, theta, eta, pr, ratings) for a in data])
    return sum([a * np.log(b) / b for (a, b) in zip(omegas, omegas.sum(-1).sum(-1))])


def prod_dist(x, theta, eta, pr):
    return (
        (theta[x[0]][:, np.newaxis, np.newaxis] * (eta[x[1], :][:, np.newaxis] * pr))
        .sum(axis=0)
        .sum(axis=0)
    )


def compute_prod_dist(data, theta, eta, pr):
    return np.array([prod_dist(a, theta, eta, pr) for a in data])
