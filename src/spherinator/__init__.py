import importlib.metadata

from spherinator import callbacks, data, distributions, models

__version__ = importlib.metadata.version("spherinator")
__all__ = [
    "callbacks",
    "data",
    "distributions",
    "models",
]
