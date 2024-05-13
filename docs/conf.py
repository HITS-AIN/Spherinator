# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Spherinator"
copyright = "2024, HITS gGmbH"
author = (
    "Kai Polsterer <kai.polsterer@h-its.org>",
    "Bernd Doser <bernd.doser@h-its.org>",
    "Andreas Fehlner <andreas.fehlner@h-its.org>",
    "Sebastian T. Gomez <sebastian.trujillogomez@h-its.org>",
)
release = "0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
