import re

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

# Single source of the version: mmsbm.__version__
with open("src/mmsbm/__init__.py", "r") as fh:
    version = re.search(r'^__version__ = "([^"]+)"', fh.read(), re.M).group(1)

setup(
    name="mmsbm",
    version=version,
    description="Compute Mixed Membership Stochastic Block Models.",
    packages=["mmsbm"],
    # Deprecated aliases of the modules of the package, kept for the whole 2.x series (removed in 3.0)
    py_modules=["expectation_maximization", "data_handler", "helpers", "backend",
                "logger", "kernels_numpy", "kernels_numba", "kernels_cupy"],
    package_dir={"": "src"},
    url="https://github.com/eudald-seeslab/mmsbm",
    author="Eudald Correig",
    author_email="eudald.correig@urv.cat",
    keywords=["bayesian analysis", "recommender systems", "network analysis", "python"],
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="BSD-3-Clause License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
            'pytest-cov',
            'coveralls',
        ],
        "numba": ["numba"],
        "cupy": ["cupy-cuda12x"],
    },
)
