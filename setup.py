from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mmsbm",
    version="0.4.1",
    description="Compute Mixed Membership Stochastic Block Models.",
    py_modules=["mmsbm", "expectation_maximization", "data_handler", "helpers", "logger"],
    package_dir={"": "src"},
    url="https://github.com/eudald-seeslab/mmsbm",
    author="Eudald Correig",
    author_email="eudald.correig@urv.cat",
    keywords=["bayesian analysis", "recommender systems", "network analysis", "python"],
    python_requires='>=3.7',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="BSD-3-Clause License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy ~= 2.0.1",
        "pandas ~= 2.2.2",
        "scikit-learn ~= 1.5.1",
        "scipy ~= 1.14.0",
        "tqdm ~= 4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
            'pytest-cov',
            'coveralls',
        ],
    },
)
