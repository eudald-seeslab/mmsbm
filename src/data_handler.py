import logging

import pandas as pd

from helpers import _invert_dict

pd.options.mode.chained_assignment = None


class DataHandler:
    obs_dict = None
    items_dict = None
    ratings_dict = None

    def __init__(self):
        pass

    @staticmethod
    def _get_data(path_):
        return pd.read_csv(path_, sep=None, usecols=[0, 1, 2], engine="python")

    @staticmethod
    def _check_data(df):
        assert df.isnull().sum().sum() == 0, "Data contains missing values. Aborting."

    @staticmethod
    def _rename_values(x, dict_):
        return [dict_[str(a)] for a in x]

    @staticmethod
    def _create_values_dict(x):
        values = sorted(set(x))
        dict_ = {}
        [dict_.update({str(b): int(a)}) for (a, b) in zip(range(len(values)), values)]

        return dict_

    def parse_train_data(self, df):
        # This is kind of weird, but the idea is that there might be more
        #  than one train/test sets and then it'll come handy
        self.obs_dict = self._create_values_dict(df.iloc[:, 0])
        self.items_dict = self._create_values_dict(df.iloc[:, 1])
        self.ratings_dict = self._create_values_dict(df.iloc[:, 2])

        df.iloc[:, 0] = self._rename_values(df.iloc[:, 0], self.obs_dict)
        df.iloc[:, 1] = self._rename_values(df.iloc[:, 1], self.items_dict)
        df.iloc[:, 2] = self._rename_values(df.iloc[:, 2], self.ratings_dict)

        # Convert whole dataframe to int
        df = df.astype(int)

        # Note that we are returning numpy arrays
        return df.values

    def parse_test_data(self, df):
        df.iloc[:, 0] = [self.obs_dict[str(a)] for a in df.iloc[:, 0]]
        df.iloc[:, 1] = [self.items_dict[str(a)] for a in df.iloc[:, 1]]
        df.iloc[:, 2] = [self.ratings_dict[str(a)] for a in df.iloc[:, 2]]

        # Convert to int to guarantee correct dtype for numpy indexing
        df = df.astype(int)

        return df.values

    @staticmethod
    def return_original_indices(x, dict_):
        return [_invert_dict(dict_)[a] for a in x]

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
            prs.update({_invert_dict(self.ratings_dict)[a]: pd.DataFrame(pr[:, :, a])})
            for a in range(pr.shape[2])
        ]

        return prs

    def format_train_data(self, data):
        # Convert to strings
        data = data.astype(str)

        self._check_data(data)

        return self.parse_train_data(data)

    def _check_test_in_train(self, data):
        test_users = set([str(a) for a in data.iloc[:, 0]])
        train_users = set(self.obs_dict.keys())

        dif = test_users.difference(train_users)
        if len(dif):
            logger = logging.getLogger("MMSBM")
            logger.warning(
                f"The observations {', '.join([str(i) for i in dif])} are in the test set but weren't in "
                f"the train set so I'll remove them."
            )

            data = data[~data.iloc[:, 0].isin([a for a in dif])]

        return data

    def format_test_data(self, data):
        # Convert to strings
        data = data.astype(str)

        # Check that it's usable
        self._check_data(data)

        # Check that all test "users" were also in train
        data = self._check_test_in_train(data)

        return self.parse_test_data(data)

    def return_dicts(self):
        return self.obs_dict, self.items_dict, self.ratings_dict
