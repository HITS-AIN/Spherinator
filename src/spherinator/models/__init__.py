"""
PyTorch Spherinator Models
"""

from importlib import import_module

from .autoencoder import Autoencoder
from .consecutive_conv_1d_layers import ConsecutiveConv1DLayer
from .consecutive_conv_2d_layers import ConsecutiveConv2DLayer
from .consecutive_conv_transpose_1d_layers import ConsecutiveConvTranspose1DLayer
from .consecutive_conv_transpose_2d_layers import ConsecutiveConvTranspose2DLayer
from .convolutional_decoder_1d import ConvolutionalDecoder1D
from .convolutional_decoder_2d import ConvolutionalDecoder2D
from .convolutional_encoder_1d import ConvolutionalEncoder1D
from .convolutional_encoder_2d import ConvolutionalEncoder2D
from .embedding_reconstruction import EmbeddingReconstruction
from .mlp import MLP
from .sequential import Sequential
from .upsampling_decoder_2d import UpsamplingDecoder2D
from .variational_autoencoder import VariationalAutoencoder
from .weights_provider import WeightsProvider
from .yaml2model import yaml2model

_LAZY_IMPORTS = {
    "HuggingFaceResNetEncoder": (".huggingface_resnet_encoder", "HuggingFaceResNetEncoder"),
    "HuggingFaceViTEncoder": (".huggingface_vit_encoder", "HuggingFaceViTEncoder"),
    "export_onnx": (".export_onnx", "export_onnx"),
    "upload_to_huggingface": (".upload_to_huggingface", "upload_to_huggingface"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "MLP",
    "Autoencoder",
    "ConsecutiveConv1DLayer",
    "ConsecutiveConv2DLayer",
    "ConsecutiveConvTranspose1DLayer",
    "ConsecutiveConvTranspose2DLayer",
    "ConvolutionalDecoder1D",
    "ConvolutionalDecoder2D",
    "ConvolutionalEncoder1D",
    "ConvolutionalEncoder2D",
    "EmbeddingReconstruction",
    "HuggingFaceResNetEncoder",
    "HuggingFaceViTEncoder",
    "Sequential",
    "UpsamplingDecoder2D",
    "VariationalAutoencoder",
    "WeightsProvider",
    "export_onnx",
    "upload_to_huggingface",
    "yaml2model",
]
