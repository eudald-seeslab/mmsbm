import numpy as np
import pandas as pd
import logging
import pytest

from data_handler import DataHandler


def test_format_train_data_pandas_stringdtype_columns():
    if not hasattr(pd, "StringDtype"):
        pytest.skip("pandas StringDtype not available")

    s = pd.StringDtype()
    train_df = pd.DataFrame(
        {
            "users": pd.Series(["u1", "u2", "u1", "u3"], dtype=s),
            "items": pd.Series(["i1", "i2", "i1", "i3"], dtype=s),
            "ratings": pd.Series(["1", "2", "3", "1"], dtype=s),
        }
    )
    dh = DataHandler()
    out = dh.format_train_data(train_df)
    assert out.shape == (4, 3)
    assert np.issubdtype(out.dtype, np.integer)


def test_format_test_data_pandas_stringdtype_after_train():
    if not hasattr(pd, "StringDtype"):
        pytest.skip("pandas StringDtype not available")

    s = pd.StringDtype()
    train_df = pd.DataFrame(
        {
            "users": pd.Series(["u1", "u2"], dtype=s),
            "items": pd.Series(["i1", "i2"], dtype=s),
            "ratings": pd.Series(["1", "2"], dtype=s),
        }
    )
    test_df = pd.DataFrame(
        {
            "users": pd.Series(["u1", "u2"], dtype=s),
            "items": pd.Series(["i1", "i2"], dtype=s),
            "ratings": pd.Series(["1", "2"], dtype=s),
        }
    )
    dh = DataHandler()
    _ = dh.format_train_data(train_df)
    out = dh.format_test_data(test_df)
    assert out.shape == (2, 3)
    assert np.issubdtype(out.dtype, np.integer)


def test_to_object_str_uses_object_dtype_columns():
    if not hasattr(pd, "StringDtype"):
        pytest.skip("pandas StringDtype not available")
    s = pd.StringDtype()
    df = pd.DataFrame(
        {
            "users": pd.Series(["u1", "u2"], dtype=s),
            "items": pd.Series(["i1", "i2"], dtype=s),
            "ratings": pd.Series(["1", "2"], dtype=s),
        }
    )
    converted = DataHandler._to_object_str(df)
    for c in converted.columns:
        assert converted[c].dtype == object


def test_test_users_subset_of_train_does_not_raise():
    # Train has users that do not appear in test; this should not raise
    train_df = pd.DataFrame(
        {
            "users": ["u1", "u2", "u3", "u1"],
            "items": ["i1", "i2", "i1", "i3"],
            "ratings": ["1", "2", "3", "1"],
        }
    )

    test_df = pd.DataFrame(
        {
            "users": ["u1", "u2"],  # subset of train users
            "items": ["i3", "i2"],
            "ratings": ["2", "1"],
        }
    )

    dh = DataHandler()

    _ = dh.format_train_data(train_df.copy())
    out = dh.format_test_data(test_df.copy())

    assert out.shape[0] == len(test_df)
    assert np.issubdtype(out.dtype, np.integer)



def test_test_users_only_in_test_are_dropped(caplog):
    # Test includes user not seen in train; that row should be removed

    caplog.set_level(logging.WARNING, logger="MMSBM")

    train_df = pd.DataFrame(
        {
            "users": ["u1", "u2"],
            "items": ["i1", "i2"],
            "ratings": ["1", "2"],
        }
    )

    test_df = pd.DataFrame(
        {
            "users": ["u3", "u1"],  # 'u3' not present in train
            "items": ["i3", "i1"],
            "ratings": ["2", "1"],
        }
    )

    dh = DataHandler()

    _ = dh.format_train_data(train_df.copy())
    out = dh.format_test_data(test_df.copy())

    # One row (u3) should be filtered out
    assert out.shape == (1, 3)
    assert np.issubdtype(out.dtype, np.integer)
    assert "u3" in caplog.text


def test_test_items_only_in_test_are_dropped(caplog):
    caplog.set_level(logging.WARNING, logger="MMSBM")

    # Test includes item not seen in train; that row should be removed
    train_df = pd.DataFrame(
        {
            "users": ["u1", "u2"],
            "items": ["i1", "i2"],
            "ratings": ["1", "2"],
        }
    )

    test_df = pd.DataFrame(
        {
            "users": ["u1", "u2"],
            "items": ["i3", "i2"],  # 'i3' not present in train
            "ratings": ["1", "2"],
        }
    )

    dh = DataHandler()

    _ = dh.format_train_data(train_df.copy())
    out = dh.format_test_data(test_df.copy())

    # One row (i3) should be filtered out
    assert out.shape == (1, 3)
    assert np.issubdtype(out.dtype, np.integer)
    assert "i3" in caplog.text


def test_test_ratings_only_in_test_are_dropped(caplog):
    caplog.set_level(logging.WARNING, logger="MMSBM")
    # Test includes rating not seen in train; that row should be removed
    train_df = pd.DataFrame(
        {
            "users": ["u1", "u2"],
            "items": ["i1", "i2"],
            "ratings": ["1", "2"],
        }
    )

    test_df = pd.DataFrame(
        {
            "users": ["u1", "u2"],
            "items": ["i1", "i2"],
            "ratings": ["3", "2"],  # '3' not present in train
        }
    )

    dh = DataHandler()

    _ = dh.format_train_data(train_df.copy())
    out = dh.format_test_data(test_df.copy())

    # One row (rating 3) should be filtered out
    assert out.shape == (1, 3)
    assert np.issubdtype(out.dtype, np.integer)
    assert "3" in caplog.text
