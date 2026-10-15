"""How the user, item and rating columns of the data are found."""
import logging

import numpy as np
import pandas as pd
import pytest

from mmsbm.data_handler import DataHandler
from mmsbm import MMSBM, MMSBMError
from tests.synthetic import planted_data
from tests.test_mmsbm import mock_data


def fitted(train, test=None, **kwargs):
    """Prediction matrix and theta of a short deterministic fit."""
    model = MMSBM(2, 2, iterations=10, seed=1, backend="numpy", log_file=None, **kwargs)
    model.fit(train, silent=True)
    prediction = model.predict(train if test is None else test)
    return model, prediction


@pytest.fixture(scope="module")
def reference():
    model, prediction = fitted(mock_data(1))
    return model.theta, prediction


def assert_same_fit(model, prediction, reference):
    theta, expected = reference
    pd.testing.assert_frame_equal(model.theta, theta)
    assert np.array_equal(prediction, expected)


@pytest.fixture
def warnings(caplog, monkeypatch):
    """``caplog`` for the MMSBM logger, which stops propagating to the root logger whenever a model sets it up."""
    import mmsbm.model as mmsbm_module

    original = mmsbm_module.setup_logger

    def propagating(name, **kwargs):
        logger = original(name, **kwargs)
        logger.propagate = True
        return logger

    monkeypatch.setattr(mmsbm_module, "setup_logger", propagating)
    logger = logging.getLogger("MMSBM")
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="MMSBM")
    yield caplog
    logger.propagate = False


def test_columns_are_found_by_name_in_any_order(reference):
    data = mock_data(1)[["ratings", "items", "users"]]
    assert_same_fit(*fitted(data), reference)


def test_other_columns_are_ignored(reference):
    data = mock_data(1)
    data.insert(0, "timestamp", np.arange(len(data)))
    data["weight"] = 1.0
    assert_same_fit(*fitted(data), reference)


def test_unnamed_three_columns_are_read_by_position_with_a_warning(reference, warnings):
    data = mock_data(1).set_axis(["a", "b", "c"], axis=1)
    assert_same_fit(*fitted(data), reference)
    assert "['a', 'b', 'c'] are taken as ['users', 'items', 'ratings'] by their position" in warnings.text


def test_unnamed_frame_with_more_columns_raises():
    data = mock_data(1).set_axis(["a", "b", "c"], axis=1)
    data["d"] = 0
    with pytest.raises(ValueError, match=r"\['users', 'items', 'ratings'\]"):
        MMSBM(2, 2, iterations=5, seed=1, log_file=None).fit(data, silent=True)


def test_partly_named_frame_raises():
    data = mock_data(1).rename(columns={"ratings": "score"})
    with pytest.raises(ValueError, match=r"\['ratings'\] are missing"):
        MMSBM(2, 2, iterations=5, seed=1, log_file=None).fit(data, silent=True)
    with pytest.raises(ValueError, match=r"\['items'\] are missing"):
        DataHandler().select_columns(mock_data(1).rename(columns={"items": "products"}), ratings="optional")


def test_configured_column_names(reference, warnings):
    data = mock_data(1).rename(columns={"users": "u", "items": "i", "ratings": "r"})[["i", "r", "u"]]
    assert_same_fit(*fitted(data, user_col="u", item_col="i", rating_col="r"), reference)
    assert warnings.text == ""


def test_configured_column_names_in_cv_fit_and_select_groups():
    data = mock_data(1)
    renamed = data.rename(columns={"users": "u", "items": "i", "ratings": "r"})[["r", "i", "u"]]
    common = dict(iterations=5, seed=3, backend="numpy", log_file=None)
    expected = MMSBM(2, 2, **common).cv_fit(data, folds=2)
    assert MMSBM(2, 2, user_col="u", item_col="i", rating_col="r", **common).cv_fit(renamed, folds=2) == expected

    expected = MMSBM(2, 2, **common).select_groups(data, [2], [2, 3], folds=2)
    grid = MMSBM(2, 2, user_col="u", item_col="i", rating_col="r", **common).select_groups(renamed, [2], [2, 3], folds=2)
    pd.testing.assert_frame_equal(grid, expected)


def test_tuples_and_arrays_are_read_by_position_without_a_warning(reference, warnings):
    data = mock_data(1)
    as_tuples = list(data.itertuples(index=False, name=None))
    assert_same_fit(*fitted(as_tuples), reference)
    assert_same_fit(*fitted(data.to_numpy()), reference)
    assert_same_fit(*fitted(data, test=as_tuples), reference)
    assert_same_fit(*fitted(data, test=data.to_numpy()), reference)
    assert warnings.text == ""


def test_predict_without_ratings(reference, warnings):
    data = mock_data(1)
    model, prediction = fitted(data, test=data[["items", "users"]])
    assert_same_fit(model, prediction, reference)
    assert model.test.shape == (len(data), 2)
    with pytest.raises(MMSBMError, match="no rating column"):
        model.score()
    with pytest.raises(MMSBMError, match="no rating column"):
        model.held_out_log_likelihood()
    assert warnings.text == ""

    # Two unnamed columns, or pairs, work too
    model, prediction = fitted(data, test=data[["users", "items"]].set_axis(["a", "b"], axis=1))
    assert_same_fit(model, prediction, reference)
    assert "['a', 'b'] are taken as ['users', 'items'] by their position" in warnings.text
    model, prediction = fitted(data, test=data[["users", "items"]].to_numpy())
    assert_same_fit(model, prediction, reference)


def test_predict_links_finds_the_pair_columns_by_name():
    data = planted_data(0, n_users=40, n_items=60, density=0.15)[0]
    model = MMSBM(2, 2, iterations=5, seed=1, backend="numpy", degree_corrected=True, log_file=None)
    model.fit(data, silent=True)
    expected = model.predict_links(data)
    assert np.array_equal(model.predict_links(data[["items", "users"]]), expected)
    assert np.array_equal(model.predict_links(data[["users", "items"]].to_numpy()), expected)
    assert np.array_equal(model.predict_links(data[["users", "items"]].set_axis(["a", "b"], axis=1)), expected)


def test_select_columns_keeps_the_configured_names():
    handler = DataHandler("u", "i", "r")
    frame = pd.DataFrame({"x": [1], "r": [3], "i": ["b"], "u": ["a"]})
    assert list(handler.select_columns(frame).columns) == ["u", "i", "r"]
    assert list(handler.select_columns(frame[["u", "i"]], ratings="optional").columns) == ["u", "i"]
    assert list(handler.select_columns([("a", "b", 3)]).columns) == ["u", "i", "r"]
    assert list(handler.select_columns([("a", "b")], ratings="optional").columns) == ["u", "i"]
    with pytest.raises(ValueError):
        handler.select_columns([("a", "b")])


def test_get_data_is_gone():
    assert not hasattr(DataHandler, "_get_data")
