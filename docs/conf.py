"""Sphinx configuration."""

from datetime import datetime

project = "Spherinator"
copyright = f"{datetime.now().year}, HITS gGmbH"
author = """Kai Polsterer <kai.polsterer@h-its.org>,
            Bernd Doser <bernd.doser@h-its.org>,
            Andreas Fehlner <andreas.fehlner@h-its.org>,
            Sebastian T. Gomez <sebastian.trujillogomez@h-its.org>"""

extensions = [
    "myst_parser",
    # "recommonmark",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
]

html_theme = "sphinx_rtd_theme"
