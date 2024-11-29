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
"""

from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_decoder_1d import ConvolutionalDecoder1D
from .convolutional_decoder_2 import ConvolutionalDecoder2
from .convolutional_decoder_224 import ConvolutionalDecoder224
from .convolutional_decoder_256 import ConvolutionalDecoder256
from .convolutional_encoder import ConvolutionalEncoder
from .convolutional_encoder_1d import ConvolutionalEncoder1D
from .convolutional_encoder_2 import ConvolutionalEncoder2
from .rotational2_autoencoder import Rotational2Autoencoder
from .rotational2_variational_autoencoder_power import (
    Rotational2VariationalAutoencoderPower,
)
from .rotational_autoencoder import RotationalAutoencoder
from .rotational_variational_autoencoder_power import (
    RotationalVariationalAutoencoderPower,
)
from .spherinator_module import SpherinatorModule

__all__ = [
    "ConvolutionalDecoder",
    "ConvolutionalDecoder2",
    "ConvolutionalDecoder224",
    "ConvolutionalDecoder256",
    "ConvolutionalEncoder",
    "ConvolutionalEncoder1D",
    "ConvolutionalDecoder1D",
    "ConvolutionalEncoder2",
    "Rotational2Autoencoder",
    "Rotational2VariationalAutoencoderPower",
    "RotationalAutoencoder",
    "RotationalVariationalAutoencoderPower",
    "SpherinatorModule",
]
