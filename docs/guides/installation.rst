Installation Guide
================

Requirements
----------

MMSBM requires:

* Python >= 3.8
* NumPy
* Pandas
* tqdm

Basic Installation
----------------

The easiest way to install MMSBM is via pip::

    pip install mmsbm

This will install MMSBM and all its dependencies.

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