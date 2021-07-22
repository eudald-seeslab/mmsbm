import os

import numpy as np
import pandas as pd


class DataHandler:
    obs_dict = None
    items_dict = None
    ratings_dict = None

    def __init__(self, data_dir, train_set, test_set):
        self.train_set = os.path.join(data_dir, train_set)
        self.test_set = os.path.join(data_dir, test_set)

    @staticmethod
    def _get_data(path_):
        return pd.read_csv(path_, sep=None, usecols=[0, 1, 2], engine="python")

    @staticmethod
    def _check_data(df):
        assert df.isnull().sum().sum() == 0, "Data contains missing values. Aborting."

    @staticmethod
    def _rename_values(x):
        values = set(x)
        dict_ = {}
        _ = [dict_.update({str(b): a}) for (a, b) in zip(range(len(values)), values)]
        return [dict_[str(a)] for a in x], dict_

    def parse_train_data(self, df):
        # This is kind of weird, but the idea is that there might be more
        #  than one train/test sets and then it'll come handy
        df.iloc[:, 0], self.obs_dict = self._rename_values(df.iloc[:, 0])
        df.iloc[:, 1], self.items_dict = self._rename_values(df.iloc[:, 1])
        df.iloc[:, 2], self.ratings_dict = self._rename_values(df.iloc[:, 2])

        # Note that we are returning numpy arrays
        return df.values

    def parse_test_data(self, df):
        df.iloc[:, 0] = [self.obs_dict[str(a)] for a in df.iloc[:, 0]]
        df.iloc[:, 1] = [self.items_dict[str(a)] for a in df.iloc[:, 1]]
        df.iloc[:, 2] = [self.ratings_dict[str(a)] for a in df.iloc[:, 2]]

        return df.values

    @staticmethod
    def _invert_dict(d):
        return {v: k for k, v in d.items()}

    def return_original_indices(self, x, dict_):
        return [self._invert_dict(dict_)[a] for a in x]

    def return_theta_indices(self, theta):
        theta = pd.DataFrame(theta)
        theta.index = self.return_original_indices(theta.index, self.obs_dict)

        return theta

    def return_eta_indices(self, eta):
        eta = pd.DataFrame(eta)
        eta.index = self.return_original_indices(eta.index, self.items_dict)

        return eta

    def return_pr_indices(self, pr):
        # I'm just creating a dict with r (num of ratings) entries containing
        #  a dataframe each and then converting the working r indices to the
        #  originals.
        prs = {}
        [
            prs.update(
                {self._invert_dict(self.ratings_dict)[a]: pd.DataFrame(pr[:, :, a])}
            )
            for a in range(pr.shape[2])
        ]

        return prs

    def import_data(self):
        # Get data
        train = self._get_data(self.train_set)
        test = self._get_data(self.test_set)
        # Check that everything is ok
        self._check_data(train)
        self._check_data(test)
        # Convert to usable indices
        return self.parse_train_data(train), self.parse_test_data(test)

    def return_dicts(self):
        return self.obs_dict, self.items_dict, self.ratings_dict


def DEPRECATED_get_data(path_):
    # Deprecated
    # TODO: remove
    _, extension = os.path.splitext(path_)
    if extension == ".csv":
        delimiter = ","
    else:
        delimiter = "\t"

    # Deliver the dataset
    return np.genfromtxt(path_, delimiter=delimiter, usecols=[0, 1, 2], dtype="int")
