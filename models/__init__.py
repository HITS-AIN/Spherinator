"""
This module is the entry point of the models package used to provide the projections.
It initializes the package and makes its modules available for import.

It contains the following modules:

1. `rotational_spherical_autoencoder`:
    A plain convolutional autoencoder projecting on a sphere with naive rotation invariance.
2. `rotational_spherical_variational_autoencoder`:
    A convolutional variational autoencoder projecting on a sphere with naive rotation invariance.
"""

from .rotational_autoencoder import RotationalAutoencoder
from .rotational_variational_autoencoder import RotationalVariationalAutoencoder
from .svae import SVAE
from .vae import VAE

__all__ = [
    'RotationalAutoencoder',
    'RotationalVariationalAutoencoder',
    'SVAE',
    'VAE',
]
