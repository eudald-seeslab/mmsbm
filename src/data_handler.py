import logging

import numpy as np
import pandas as pd

from helpers import _invert_dict

pd.options.mode.chained_assignment = None


class DataHandler:
    """Encodes the user, item and rating labels of the data as integer indices.

    Parameters
    ----------
    user_col, item_col, rating_col : str
        Names of the columns that hold the users, the items and the ratings.
        See :meth:`select_columns` for how they are looked up.
    """

    ROLES = ("users", "items", "ratings")

    obs_dict = None
    items_dict = None
    ratings_dict = None

    def __init__(self, user_col="users", item_col="items", rating_col="ratings"):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

    @property
    def column_names(self):
        return [self.user_col, self.item_col, self.rating_col]

    def select_columns(self, data, ratings="required"):
        """Return the user, item and rating columns of ``data``, in that order.

        The columns are looked up by name (``user_col``, ``item_col`` and
        ``rating_col``), so their order in ``data`` does not matter and any
        other column is ignored. A DataFrame that has none of the names and
        exactly three columns (two, when the ratings are optional) is read by
        position, with a warning. Anything that is not a DataFrame (a list of
        tuples, a 2-D array) is read by position without a warning, since it
        carries no names. Any other case raises ``ValueError``.

        Parameters
        ----------
        data : DataFrame or array-like of shape (n, 3) or (n, 2)
        ratings : {"required", "optional"}
            Whether the rating column must be present. When it is optional
            and absent, the result has two columns.

        Returns
        -------
        DataFrame
            Columns ``[user_col, item_col, rating_col]`` or
            ``[user_col, item_col]``.
        """
        assert ratings in ("required", "optional"), "ratings must be 'required' or 'optional'."
        is_frame = isinstance(data, pd.DataFrame)
        if not is_frame:
            data = pd.DataFrame(data)

        names = self.column_names
        found = [name in data.columns for name in names]
        needed = names if ratings == "required" else names[:2]

        if all(name in data.columns for name in needed):
            keep = names if found[2] else names[:2]
            return data[keep]

        widths = (3,) if ratings == "required" else (3, 2)
        if not any(found) and data.shape[1] in widths:
            keep = names[: data.shape[1]]
            if is_frame:
                logging.getLogger("MMSBM").warning(
                    f"None of the columns {names} is in the data, so the columns {list(data.columns)} are taken as "
                    f"{list(self.ROLES[: len(keep)])} by their position. Name them, or pass user_col, item_col and "
                    f"rating_col to MMSBM."
                )
            return data.set_axis(keep, axis=1)

        missing = [name for name in needed if name not in data.columns]
        raise ValueError(
            f"The data must have the columns {needed} (user_col, item_col and rating_col); {missing} are missing "
            f"and the data has the columns {list(data.columns)}. A DataFrame without any of the names is read by "
            f"position only when it has exactly {' or '.join(str(w) for w in widths)} columns."
        )

    @staticmethod
    def _check_data(df):
        assert df.isnull().sum().sum() == 0, "Data contains missing values. Aborting."

    @staticmethod
    def _to_object_str(data):
        out = data.copy()
        for col in out.columns:
            values = [str(x) for x in out[col].tolist()]
            out[col] = pd.Series(values, index=out.index, dtype=object)
        return out

    @staticmethod
    def _rename_values(x, dict_):
        return [dict_[str(a)] for a in x]

    @staticmethod
    def _create_values_dict(x):
        values = sorted(set(x))
        dict_ = {}
        [dict_.update({str(b): int(a)}) for (a, b) in zip(range(len(values)), values)]

        return dict_

    def _dicts(self):
        return (self.obs_dict, self.items_dict, self.ratings_dict)

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
        """Encode a frame of ``[users, items]`` or ``[users, items, ratings]`` columns."""
        for position, dict_ in zip(range(df.shape[1]), self._dicts()):
            df.iloc[:, position] = self._rename_values(df.iloc[:, position], dict_)

        # Convert to int to guarantee correct dtype for numpy indexing
        df = df.astype(int)

        return df.values

    @staticmethod
    def return_original_indices(x, dict_):
        inverse = _invert_dict(dict_)
        return [inverse[a] for a in x]

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
        inverse = _invert_dict(self.ratings_dict)
        return {inverse[a]: pd.DataFrame(pr[:, :, a]) for a in range(pr.shape[2])}

    def format_train_data(self, data):
        data = self.select_columns(data)

        # Convert to strings
        data = self._to_object_str(data)

        self._check_data(data)

        return self.parse_train_data(data)

    def _check_test_in_train(self, data):
        """Drop the rows whose user, item or rating (the columns present) was not in the training data."""
        logger = logging.getLogger("MMSBM")

        for position, (role, train_dict) in enumerate(zip(self.ROLES, self._dicts())):
            if position >= data.shape[1]:
                break
            values = data.iloc[:, position]
            test_values = set(str(a) for a in values)
            train_values = set(train_dict.keys())
            dif_values = test_values.difference(train_values)
            if len(dif_values):
                logger.warning(
                    f"The {role} {', '.join(str(a) for a in dif_values)} are in the test set but weren't in "
                    f"the train set so I'll remove them."
                )
                data = data[~values.isin(dif_values).to_numpy()]

        return data

    def format_test_data(self, data):
        """Encode test data; the rating column is optional.

        Returns an array of shape (n, 3), or (n, 2) when ``data`` has no
        rating column, in which case the predictions cannot be scored.
        """
        data = self.select_columns(data, ratings="optional")

        # Convert to strings
        data = self._to_object_str(data)

        # Check that it's usable
        self._check_data(data)

        # Check that all test "users" were also in train
        data = self._check_test_in_train(data)

        return self.parse_test_data(data)

    def format_pairs(self, data):
        """Encode (user, item) pairs with the training dictionaries.

        Only the user and item columns are used, so a ratings column is
        optional and ignored.  Pairs whose user or item was not seen in
        training are dropped with a warning, as in :meth:`format_test_data`.

        Returns
        -------
        ndarray of shape (N, 2)
            Encoded ``[user_idx, item_idx]`` rows.
        """
        data = self.select_columns(data, ratings="optional").iloc[:, :2]
        data = self._to_object_str(data)
        self._check_data(data)
        data = self._check_test_in_train(data)

        return self.parse_test_data(data).reshape(-1, 2)

    def return_dicts(self):
        return self.obs_dict, self.items_dict, self.ratings_dict
