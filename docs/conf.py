"""Sphinx configuration."""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "Spherinator"
copyright = f"{datetime.now().year}, HITS gGmbH"
author = """Kai Polsterer <kai.polsterer@h-its.org>,
            Bernd Doser <bernd.doser@h-its.org>,
            Andreas Fehlner <andreas.fehlner@h-its.org>,
            Sebastian T. Gomez <sebastian.trujillogomez@h-its.org>"""

extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

autodoc_typehints = "description"  # or "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
bibtex_bibfiles = ["references.bib"]
