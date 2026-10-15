"""Mixed-membership stochastic block models for recommendation.

The public API is :class:`MMSBM`. The other modules of the package (:mod:`mmsbm.data_handler`,
:mod:`mmsbm.expectation_maximization`, :mod:`mmsbm.helpers`, :mod:`mmsbm.backend`, :mod:`mmsbm.logger`,
the ``mmsbm.kernels_*`` modules and :mod:`mmsbm.exceptions`) are its building blocks.
"""
from importlib import import_module

__version__ = "2.0.1"

from mmsbm._deprecation import warn_deprecated
from mmsbm.data_handler import DataHandler
from mmsbm.exceptions import MMSBMError, NotFittedError
from mmsbm.expectation_maximization import ExpectationMaximization
from mmsbm.model import MMSBM

# The class is public as ``mmsbm.MMSBM`` (also the path stored in pickles, as in 1.x).
MMSBM.__module__ = __name__

__all__ = ["MMSBM", "DataHandler", "ExpectationMaximization", "MMSBMError", "NotFittedError", "__version__"]

# Names that the module ``mmsbm`` of 1.x exposed because it imported them.
_MOVED = {
    "assign_folds": "mmsbm.helpers",
    "drop_unseen": "mmsbm.helpers",
    "setup_logger": "mmsbm.logger",
}


def __getattr__(name):
    if name in _MOVED:
        warn_deprecated(f"'mmsbm.{name}'", f"'{_MOVED[name]}.{name}'")
        return getattr(import_module(_MOVED[name]), name)
    raise AttributeError(f"module 'mmsbm' has no attribute '{name}'")
