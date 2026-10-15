API reference
=============

``mmsbm`` is a package since 2.0. ``from mmsbm import MMSBM`` is all that
most programs need; the package also exports ``DataHandler``,
``ExpectationMaximization``, the exceptions and ``__version__``. The old
top-level modules (``helpers``, ``logger``, ``backend``, ``data_handler``,
``expectation_maximization``, ``kernels_*``) are deprecated aliases of the
modules below and are removed in 3.0.

The model
---------

.. autoclass:: mmsbm.MMSBM
   :members:

Exceptions
----------

.. automodule:: mmsbm.exceptions
   :members:

Data handling
-------------

.. automodule:: mmsbm.data_handler
   :members:

.. automodule:: mmsbm.helpers
   :members:

Expectation-maximization
------------------------

.. automodule:: mmsbm.expectation_maximization
   :members:

.. automodule:: mmsbm.backend
   :members:

Kernels
-------

.. automodule:: mmsbm.kernels_numpy
   :members:

.. automodule:: mmsbm.kernels_numba
   :members:

.. automodule:: mmsbm.kernels_cupy
   :members:

Logging
-------

.. automodule:: mmsbm.logger
   :members:
