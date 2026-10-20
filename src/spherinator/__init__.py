import importlib.metadata

__version__ = importlib.metadata.version("spherinator")


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module


__all__ = [
    "callbacks",
    "data",
    "distributions",
    "losses",
    "models",
]
