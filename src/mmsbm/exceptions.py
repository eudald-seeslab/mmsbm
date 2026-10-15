"""Exceptions raised by mmsbm.

Until 2.0 these conditions raised ``AssertionError`` from ``assert`` statements, which ``python -O`` removes.
For the whole 2.x series the classes inherit from ``AssertionError`` as well as from ``ValueError``, so that
``except AssertionError`` keeps catching them; the ``AssertionError`` base is dropped in 3.0.
"""


class MMSBMError(ValueError, AssertionError):
    """Invalid input, or an operation that the state of the model does not allow."""


class NotFittedError(MMSBMError, AttributeError):
    """A method needs :meth:`MMSBM.fit` (or :meth:`MMSBM.predict`) to have been called first."""
