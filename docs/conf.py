# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "Modeling HIV with an Open Cohort SEIR-S Framework"
copyright = "2024, Ryan O'Dea, Emily Stewart, Patrick Thornton"
author = "Ryan O'Dea, Emily Stewart, Patrick Thornton"
release = "1.0.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# HTML output options
html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']
