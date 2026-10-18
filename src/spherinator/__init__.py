import importlib.metadata

__version__ = importlib.metadata.version("spherinator")
__all__ = [
    "callbacks",
    "data",
    "distributions",
    "losses",
    "models",
]
