"""The compatibility paths of the 2.x series and their warnings."""
import importlib
import pickle
import sys
import warnings

import numpy as np
import pytest

import mmsbm
from mmsbm import MMSBM, MMSBMError, NotFittedError
from tests.test_mmsbm import mock_data

OLD_MODULES = [
    "helpers", "logger", "backend", "data_handler", "expectation_maximization",
    "kernels_numpy", "kernels_numba", "kernels_cupy",
]


def import_afresh(name):
    """Import a module again so that the warning of its first import fires."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize("name", OLD_MODULES)
def test_old_module_paths_warn_and_alias_the_package(name):
    try:
        new = importlib.import_module(f"mmsbm.{name}")
    except ImportError:
        pytest.skip(f"mmsbm.{name} cannot be imported on this machine")
    with pytest.warns(FutureWarning, match=f"The module '{name}' is deprecated.*use mmsbm.{name} instead") as record:
        old = import_afresh(name)
    public = [k for k in vars(new) if not (k.startswith("__") and k.endswith("__"))]
    assert public
    assert all(getattr(old, k) is getattr(new, k) for k in public)
    assert record[0].filename == __file__, "the warning must point at the importer"


def test_moved_names_of_the_package_warn():
    with pytest.warns(FutureWarning, match="'mmsbm.assign_folds' is deprecated.*'mmsbm.helpers.assign_folds'") as record:
        assign_folds = mmsbm.assign_folds
    assert assign_folds is mmsbm.helpers.assign_folds
    assert record[0].filename == __file__
    with pytest.warns(FutureWarning, match="'mmsbm.setup_logger'"):
        assert mmsbm.setup_logger is mmsbm.logger.setup_logger
    with pytest.raises(AttributeError):
        mmsbm.no_such_name


def test_public_api():
    assert MMSBM.__module__ == "mmsbm"
    assert set(mmsbm.__all__) == {"MMSBM", "DataHandler", "ExpectationMaximization", "MMSBMError", "NotFittedError", "__version__"}
    assert mmsbm.DataHandler is mmsbm.data_handler.DataHandler
    assert mmsbm.ExpectationMaximization is mmsbm.expectation_maximization.ExpectationMaximization


@pytest.fixture(scope="module")
def model():
    model = MMSBM(2, 2, iterations=5, seed=1, backend="numpy")
    model.fit(mock_data(1), silent=True)
    model.predict(mock_data(2))
    return model


def test_rats_argument_of_choose_best_run_warns(model):
    with pytest.warns(FutureWarning, match="'rats' argument of 'choose_best_run'"):
        assert model.choose_best_run(np.zeros(1)) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert model.choose_best_run() == 0


def test_d0_and_d1_warn(model):
    with pytest.warns(FutureWarning, match="'MMSBM.d0' is deprecated"):
        d0 = model.d0
    with pytest.warns(FutureWarning, match="'MMSBM.d1' is deprecated"):
        d1 = model.d1
    assert set(d0) == set(model.train[:, 0])
    assert set(d1) == set(model.train[:, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(NotFittedError):
            MMSBM(2, 2).d0


def test_errors_are_value_errors_and_still_assertion_errors():
    unfitted = MMSBM(2, 2, iterations=5, seed=1)
    with pytest.raises(NotFittedError) as info:
        unfitted.predict(mock_data(1))
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, AttributeError)
    assert isinstance(info.value, AssertionError), "1.x code catches AssertionError"

    with pytest.raises(AssertionError):
        unfitted.score()
    with pytest.raises(ValueError):
        unfitted.held_out_log_likelihood()

    with pytest.raises(MMSBMError) as info:
        unfitted.cv_fit(mock_data(1), folds=1)
    assert not isinstance(info.value, NotFittedError)
    with pytest.raises(MMSBMError, match="missing values"):
        unfitted.fit(mock_data(1).assign(ratings=[None] * 100), silent=True)


def test_pickles_of_1x_models_load(model):
    """1.x pickles refer to ``mmsbm.MMSBM``, ``data_handler.DataHandler`` and ``expectation_maximization.ExpectationMaximization``."""
    old_paths = {
        mmsbm.data_handler.DataHandler: "data_handler",
        mmsbm.expectation_maximization.ExpectationMaximization: "expectation_maximization",
    }
    current = {cls: cls.__module__ for cls in old_paths}
    try:
        for cls, module in old_paths.items():
            cls.__module__ = module
        payload = pickle.dumps(model)
    finally:
        for cls, module in current.items():
            cls.__module__ = module
    assert b"mmsbm.data_handler" not in payload and b"mmsbm.expectation_maximization" not in payload

    for name in old_paths.values():
        sys.modules.pop(name, None)
    with pytest.warns(FutureWarning, match="The module '(data_handler|expectation_maximization)' is deprecated"):
        loaded = pickle.loads(payload)
    assert type(loaded) is MMSBM
    assert type(loaded.data_handler) is mmsbm.data_handler.DataHandler
    assert np.array_equal(loaded.predict(mock_data(2)), model.prediction_matrix)


def test_pickles_use_the_public_path(model):
    payload = pickle.dumps(model)
    assert b"mmsbm.model" not in payload
    loaded = pickle.loads(payload)
    assert np.array_equal(loaded.predict(mock_data(2)), model.prediction_matrix)
