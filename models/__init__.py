"""
This module is the entry point of the models package used to provide the projections.
It initializes the package and makes its modules available for import.

It contains the following modules:

1. `rotational_spherical_projecting_autoencoder`:
    A plain convolutional autoencoder projecting on a sphere with naive rotation invariance.
2. `simple`: TODO
"""

from .rotational_spherical_projecting_autoencoder import RotationalSphericalProjectingAutoencoder
from .simple import SimpleModel
from .vae import VAE

__all__ = [
    'RotationalSphericalProjectingAutoencoder',
    'SimpleModel',
    'VAE'
]
