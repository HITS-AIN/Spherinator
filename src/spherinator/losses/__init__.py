"""
This module contains the loss functions used in Spherinator.
"""

from importlib import import_module

from .combined_loss import CombinedLoss


def __getattr__(name: str):
    if name != "PerceptualLoss":
        raise AttributeError(name)
    value = import_module(".perceptional_loss", __name__).PerceptualLoss
    globals()[name] = value
    return value


__all__ = [
    "CombinedLoss",
    "PerceptualLoss",
]
