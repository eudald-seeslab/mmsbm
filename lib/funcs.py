import numpy as np
import pandas as pd
from numba import jit


def init_random_array(shape, rng):
    return rng.random(shape)


@jit
def update_coefs(data, ratings, theta, eta, pr):
    # Initialize the empty objects
    n_theta, n_eta, n_pr = np.zeros_like(theta), np.zeros_like(eta), np.zeros_like(pr)

    for x in data:
        ra = ratings.index(x[2])
        omega = (theta[x[0]][:, np.newaxis] * (eta[x[1], :] * pr[:, :, ra])).sum().sum()
        a = (theta[x[0]][:, np.newaxis] * (eta[x[1], :] * pr[:, :, ra])) / omega
        n_theta[x[0], :] += a.sum(1)
        n_eta[x[1], :] += a.sum(0)
        n_pr[:, :, ra] += a

    return n_theta, n_eta, n_pr


def normalize_with_d(df, d):
    return df / [np.repeat(max(len(a), 1), df.shape[1]) for a in list(d.values())]


def normalize_with_self(df):
    # This function should generalize better (TODO)
    temp = df.reshape((df.shape[0] * df.shape[1], df.shape[2]))
    return (temp / (np.where(temp.sum(axis=1) == 0, 1, temp.sum(axis=1)))[:, np.newaxis]).reshape(df.shape)


@jit
def compute_likelihood(data, ratings, theta, eta, pr):
    likelihood = 0
    for x in data:
        ra = ratings.index(x[2])
        omega = (theta[x[0]][:, np.newaxis] * (eta[x[1], :] * pr[:, :, ra])).sum().sum()
        likelihood += ((theta[x[0]][:, np.newaxis] * (eta[x[1], :] * pr[:, :, ra])) * np.log(omega) / omega).sum().sum()

    return likelihood


def prod_dist(x, theta, eta, pr, sampling):
    return (theta[x[0]][:, np.newaxis, np.newaxis] * (eta[x[1], :][:, np.newaxis] * pr)).sum(axis=0).sum(axis=0) / sampling


# Note: not jit'd because numba sucks
def compute_prod_dist(data, theta, eta, pr, sampling):
    return np.apply_along_axis(prod_dist, axis=1, arr=data, theta=theta, eta=eta, pr=pr, sampling=sampling)


def weighting(x, ratings):
    # TODO: this is likely improvable
    return sum([a * b for (a, b) in zip(x, ratings)])


def compute_indicators(rat, test, ratings):
    rat = pd.DataFrame(rat)

    # Note the + 1
    rat["pred"] = np.argmax(rat.values, axis=1) + 1

    # Add the real results
    rat = rat.assign(real=pd.Series(test[:, 2]))

    # Remove observations without predictions
    rat = rat.loc[rat.iloc[:, :len(ratings)].sum(axis=1) != 0, :]

    # Check the ones we got right
    rat["true"] = np.where(rat["pred"] == rat["real"], 1, 0)

    # squared error (which is not squared error but ok)
    rat["s2"] = abs(rat["pred"] - rat["real"])

    # Same but weighed
    # Note that we are assuming that weights are the first R columns
    rat["pred_pond"] = [weighting(a, ratings) for a in rat.iloc[:, :len(ratings)].values]
    rat["true_pond"] = np.where(rat["real"] == round(rat["pred_pond"]), 1, 0)
    rat["s2pond"] = abs(rat["pred_pond"] - rat["real"])

    return rat


def compute_final_stats(rat):
    # Final model quality indicators
    accuracy = rat["true"].sum() / rat.shape[0]
    mae = 1 - rat["true_pond"].sum() / rat.shape[0]

    # Errors
    s2 = rat["s2"].sum()
    s2pond = rat["s2pond"].sum()

    return accuracy, mae, s2, s2pond