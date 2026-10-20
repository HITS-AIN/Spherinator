"""
PyTorch Lightning callbacks
"""

from importlib import import_module

from .kl_annealing import KLAnnealing
from .param_manager import ParamManager


def __getattr__(name: str):
    if name != "LogReconstructionCallback":
        raise AttributeError(name)
    value = import_module(".log_reconstruction_callback", __name__).LogReconstructionCallback
    globals()[name] = value
    return value


__all__ = [
    "KLAnnealing",
    "LogReconstructionCallback",
    "ParamManager",
]
