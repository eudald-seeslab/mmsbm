Installation Guide
================

.. _installation:

Requirements
------------

MMSBM requires:

* Python >= 3.7
* NumPy
* tqdm

Basic Installation
------------------

The easiest way to install the base ``mmsbm`` package is via pip::

    pip install mmsbm

This will install the package with the default ``numpy`` backend.

Installing Optional Backends
----------------------------

For enhanced performance, you can install optional dependencies for accelerated backends.

Numba Backend (CPU JIT Compilation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable the ``numba`` backend, install the Numba dependency::

    pip install mmsbm[numba]


CuPy Backend (GPU Acceleration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable the ``cupy`` backend for NVIDIA GPUs, you first need a compatible CUDA toolkit installed on your system. Then, you can install the required dependency::

    pip install mmsbm[cupy]

.. note::
   The CuPy backend requires **Python 3.12 or lower**. As of December 2025, CuPy does not provide pre-built wheels for Python 3.13+.

You can also install all extras at once::

    pip install mmsbm[numba,cupy]

Development Installation
---------------------

For development, you can install MMSBM from source::

    git clone https://github.com/eudald-seeslab/mmsbm.git
    cd mmsbm
    pip install -e ".[dev]"

This will install additional dependencies needed for development:

* pytest for testing
* sphinx for documentation
* black for code formatting

Verifying Installation
-------------------

To verify the installation, you can run::

    python -c "import mmsbm; print(mmsbm.__version__)"

Running Tests
-----------

To run the test suite::

    python -m pytest tests/

Building Documentation
-------------------

To build the documentation locally::

    cd docs
    make html

The documentation will be available in ``docs/_build/html/``.

Troubleshooting
-------------

Common Issues
^^^^^^^^^^^

1. **ImportError: No module named 'mmsbm'**

   Make sure you've installed the package correctly::

       pip install --upgrade mmsbm

2. **Version conflicts**

   Try creating a new virtual environment::

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       pip install mmsbm

Getting Help
----------

If you encounter any issues:

1. Check the :doc:`troubleshooting guide <../troubleshooting>`
2. Search existing GitHub issues
3. Open a new GitHub issue with:

   * Your Python version
   * MMSBM version
   * Complete error traceback
   * Minimal example reproducing the issue