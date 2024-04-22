"""
This module is the entry point of the models package used to provide the projections.
It initializes the package and makes its modules available for import.

It contains the following modules:

1. `RotationalAutoencoder`:
    A plain convolutional autoencoder projecting on a sphere with naive rotation invariance.
2. `RotationalVariationalAutoencoder`:
    A convolutional variational autoencoder projecting on a sphere with naive rotation invariance.
3. `RotationalVariationalAutoencoderPower`:
    A convolutional variational autoencoder using power spherical distribution.
4. `RotationalVariationalAutoencoderMMD`:
    A convolutional variational autoencoder (same as 2.) using MMD loss.
5. `RotationalVariationalAutoencoderPowerMMD`:
    A convolutional variational autoencoder using power spherical distribution and MMD loss.
"""

from .rotational_autoencoder import RotationalAutoencoder
from .rotational_variational_autoencoder import RotationalVariationalAutoencoder
from .rotational_variational_autoencoder_power import RotationalVariationalAutoencoderPower
from .rotational_variational_autoencoder_mmd import RotationalVariationalAutoencoderMMD
from .rotational_variational_autoencoder_power_mmd import RotationalVariationalAutoencoderPowerMMD

__all__ = [
    'RotationalAutoencoder',
    'RotationalVariationalAutoencoder',
    'RotationalVariationalAutoencoderPower',
    'RotationalVariationalAutoencoderMMD'
    'RotationalVariationalAutoencoderPowerMMD'
]
