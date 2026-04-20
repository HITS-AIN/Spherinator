import pytest
import torch

from spherinator.models import HuggingFaceResNetEncoder, HuggingFaceViTEncoder


@pytest.mark.parametrize("output_dim", [None, 64])
def test_resnet_encoder_output_shape(output_dim):
    encoder = HuggingFaceResNetEncoder(output_dim=output_dim)
    x = torch.randn(2, 3, 128, 128)
    out = encoder(x)
    expected_dim = output_dim if output_dim is not None else 512  # resnet-18 hidden_sizes[-1]
    assert out.shape == (2, expected_dim)


def test_resnet_encoder_freeze():
    encoder = HuggingFaceResNetEncoder(freeze=True)
    for param in encoder.resnet.parameters():
        assert not param.requires_grad


def test_resnet_encoder_example_input():
    encoder = HuggingFaceResNetEncoder()
    assert encoder.example_input_array.shape == (1, 3, 128, 128)


@pytest.mark.parametrize("output_dim", [None, 64])
def test_vit_encoder_output_shape(output_dim):
    encoder = HuggingFaceViTEncoder(output_dim=output_dim)
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    expected_dim = output_dim if output_dim is not None else 768  # vit-base hidden_size
    assert out.shape == (2, expected_dim)


def test_vit_encoder_freeze():
    encoder = HuggingFaceViTEncoder(freeze=True)
    for param in encoder.vit.parameters():
        assert not param.requires_grad


def test_vit_encoder_example_input():
    encoder = HuggingFaceViTEncoder()
    assert encoder.example_input_array.shape == (1, 3, 224, 224)
