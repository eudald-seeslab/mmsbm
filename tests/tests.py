import os
from unittest import TestCase
import numpy as np

from lib.funcs import compute_likelihood, compute_prod_dist


class TestFuncs(TestCase):

    sampling = 1

    def setUp(self) -> None:
        # Get the data we'll need
        wd = os.path.join(os.getcwd(), "..")
        self.theta = np.genfromtxt(os.path.join(wd, "fixtures", "theta.csv"), delimiter=",")[1:, 1:]
        self.eta = np.genfromtxt(os.path.join(wd, "fixtures", "eta.csv"), delimiter=",")[1:, 1:]
        pr0 = np.genfromtxt(os.path.join(wd, "fixtures", "pr0.csv"), delimiter=",")[1:, 1:]
        pr1 = np.genfromtxt(os.path.join(wd, "fixtures", "pr1.csv"), delimiter=",")[1:, 1:]
        pr2 = np.genfromtxt(os.path.join(wd, "fixtures", "pr2.csv"), delimiter=",")[1:, 1:]
        self.pr = np.swapaxes(np.swapaxes(np.dstack((pr0, pr1, pr2)), 1, 2), 0, 1)
        self.train = np.genfromtxt(os.path.join(wd, "fixtures", "train.csv"), delimiter=",", dtype="int")[1:, 1:]
        self.test = np.genfromtxt(os.path.join(wd, "fixtures", "test.csv"), delimiter=",", dtype="int")[1:, 1:]
        self.ratings = list(range(1, 6))
        self.rat = np.genfromtxt(os.path.join(wd, "fixtures", "rat.csv"), delimiter=",")[1:, 1:]

    def test_likelihood(self):
        np.testing.assert_almost_equal(
            sum(compute_likelihood(self.train, self.ratings, self.theta, self.eta, self.pr)), -105621.674244
        )

    def test_prod_dist(self):
        np.testing.assert_almost_equal(
            compute_prod_dist(self.test, self.theta, self.eta, self.pr), self.rat
        )
