Installation Guide
==================

.. _installation:

Requirements
------------

MMSBM requires:

* Python >= 3.10
* NumPy >= 1.26
* pandas >= 2.2
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

To enable the ``cupy`` backend for NVIDIA GPUs, you first need a compatible NVIDIA driver and CUDA toolkit installed on your system. ``mmsbm[cupy]`` installs ``cupy-cuda12x``, the CuPy wheel built for CUDA 12, which is the most common toolkit::

    pip install mmsbm[cupy]

.. note::
   The CuPy wheel must match the CUDA toolkit of your machine. With CUDA 13 or CUDA 11 install ``cupy-cuda13x`` or ``cupy-cuda11x`` yourself instead of the extra (only one of the three can be installed in an environment); ``mmsbm`` uses whichever is present. See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for the toolkit setup. CuPy 14 provides wheels for Python 3.10 to 3.14.

You can also install all extras at once::

    pip install mmsbm[numba,cupy]

Development Installation
------------------------

For development, you can install MMSBM from source::

    git clone https://github.com/eudald-seeslab/mmsbm.git
    cd mmsbm
    pip install -e ".[dev]"

This will install additional dependencies needed for development:

* pytest for testing
* sphinx for documentation
* black for code formatting

Verifying Installation
----------------------

To verify the installation, you can run::

    python -c "import mmsbm; print(mmsbm.__version__)"

Running Tests
-------------

To run the test suite::

    python -m pytest tests/

Building Documentation
----------------------

To build the documentation locally::

    cd docs
    make html

The documentation will be available in ``docs/_build/html/``.

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

1. **ImportError: No module named 'mmsbm'**

   Make sure you've installed the package correctly::

       pip install --upgrade mmsbm

2. **Version conflicts**

   Try creating a new virtual environment::

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       pip install mmsbm

Getting Help
------------

If you encounter any issues:

1. Check the :doc:`troubleshooting guide <../troubleshooting>`
2. Search existing GitHub issues
3. Open a new GitHub issue with:

   * Your Python version
   * MMSBM version
   * Complete error traceback
   * Minimal example reproducing the issue