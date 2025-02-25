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

from .autoencoder import Autoencoder
from .autoencoder_pure import AutoencoderPure
from .consecutive_conv_1d_layers import ConsecutiveConv1DLayer
from .consecutive_conv_transpose_1d_layers import ConsecutiveConvTranspose1DLayer
from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_decoder_1d import ConvolutionalDecoder1D
from .convolutional_decoder_1d_gen import ConvolutionalDecoder1DGen
from .convolutional_decoder_2 import ConvolutionalDecoder2
from .convolutional_decoder_224 import ConvolutionalDecoder224
from .convolutional_decoder_256 import ConvolutionalDecoder256
from .convolutional_encoder import ConvolutionalEncoder
from .convolutional_encoder_1d import ConvolutionalEncoder1D
from .convolutional_encoder_1d_gen import ConvolutionalEncoder1DGen
from .convolutional_encoder_2 import ConvolutionalEncoder2
from .dense_model import DenseModel
from .embedding_reconstruction import EmbeddingReconstruction
from .rotational2_autoencoder import Rotational2Autoencoder
from .rotational2_variational_autoencoder_power import (
    Rotational2VariationalAutoencoderPower,
)
from .rotational_autoencoder import RotationalAutoencoder
from .rotational_variational_autoencoder_power import (
    RotationalVariationalAutoencoderPower,
)
from .spherinator_module import SpherinatorModule
from .variational_autoencoder import VariationalAutoencoder
from .variational_autoencoder_pure import VariationalAutoencoderPure
from .weights_provider import WeightsProvider

__all__ = [
    "Autoencoder",
    "AutoencoderPure",
    "ConsecutiveConv1DLayer",
    "ConsecutiveConvTranspose1DLayer",
    "ConvolutionalDecoder",
    "ConvolutionalDecoder1D",
    "ConvolutionalDecoder1DGen",
    "ConvolutionalDecoder2",
    "ConvolutionalDecoder224",
    "ConvolutionalDecoder256",
    "ConvolutionalEncoder",
    "ConvolutionalEncoder1D",
    "ConvolutionalEncoder1DGen",
    "ConvolutionalEncoder2",
    "DenseModel",
    "EmbeddingReconstruction",
    "Rotational2Autoencoder",
    "Rotational2VariationalAutoencoderPower",
    "RotationalAutoencoder",
    "RotationalVariationalAutoencoderPower",
    "SpherinatorModule",
    "VariationalAutoencoder",
    "VariationalAutoencoderPure",
    "WeightsProvider",
]
