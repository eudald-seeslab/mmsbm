from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mmsbm",
    version="0.0.4",
    description="Compute Mixed Membership Stochastic Block Models.",
    py_modules=["mmsbm"],
    package_dir={"": "mmsbm"},
    url="https://github.com/eudald-seeslab/mmsbm",
    author="Eudald Correig",
    author_email="eudald.correig@urv.cat",
    keywords=["bayesian analysis", "recommender systems", "network analysis", "python"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy ~= 1.20.3",
        "pandas ~= 1.2.4",
        "scikit-learn ~= 0.24.2",
        "scipy ~= 1.6.3",
        "tqdm ~= 4.61.0",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
        ],
    }
)
