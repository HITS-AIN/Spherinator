"""
This module is the entry point of the models package used to provide the projections.
It initializes the package and makes its modules available for import.

It contains the following modules:

1. `rotational_spherical_projecting_autoencoder`:
    A plain convolutional autoencoder projecting on a sphere with naive rotation invariance.
2. `simple`: TODO
"""

from .rotational_spherical_autoencoder import RotationalSphericalAutoencoder
from .rotational_spherical_variational_autoencoder import \
    RotationalSphericalVariationalAutoencoder
from .simple import SimpleModel
from .svae import SVAE
from .vae import VAE

__all__ = [
    'RotationalSphericalAutoencoder',
    'RotationalSphericalVariationalAutoencoder',
    'SimpleModel',
    'SVAE',
    'VAE',
]
