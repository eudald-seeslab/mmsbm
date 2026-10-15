# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import re
from pathlib import Path

project = 'mmsbm'
copyright = '2024, Eudald Correig-Fraga'
author = 'Eudald Correig-Fraga'
# Single source of the version: mmsbm.__version__
release = re.search(
    r'^__version__ = "([^"]+)"',
    (Path(__file__).resolve().parents[1] / 'src' / 'mmsbm.py').read_text(encoding='utf-8'),
    re.M,
).group(1)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Math settings
mathjax3_config = {
    'tex': {
        'macros': {
            'vec': [r'\boldsymbol{#1}', 1],
            'mat': [r'\boldsymbol{#1}', 1],
        }
    }
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'testing.ipynb']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
