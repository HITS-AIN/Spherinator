import pytest
import torch
from transformers import ResNetConfig, ResNetModel, ViTConfig, ViTModel

from spherinator.models import HuggingFaceResNetEncoder, HuggingFaceViTEncoder


@pytest.fixture(autouse=True)
def local_pretrained_models(monkeypatch):
    resnet = ResNetModel(ResNetConfig(hidden_sizes=[8, 16, 24, 32], depths=[1, 1, 1, 1]))
    vit = ViTModel(
        ViTConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )
    monkeypatch.setattr(ResNetModel, "from_pretrained", lambda *_args, **_kwargs: resnet)
    monkeypatch.setattr(ViTModel, "from_pretrained", lambda *_args, **_kwargs: vit)


@pytest.mark.parametrize("output_dim", [None, 64])
def test_resnet_encoder_output_shape(output_dim):
    encoder = HuggingFaceResNetEncoder(output_dim=output_dim)
    x = torch.randn(2, 3, 128, 128)
    out = encoder(x)
    expected_dim = output_dim if output_dim is not None else 32
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
    x = torch.randn(2, 3, 32, 32)
    out = encoder(x)
    expected_dim = output_dim if output_dim is not None else 32
    assert out.shape == (2, expected_dim)


def test_vit_encoder_freeze():
    encoder = HuggingFaceViTEncoder(freeze=True)
    for param in encoder.vit.parameters():
        assert not param.requires_grad


def test_vit_encoder_example_input():
    encoder = HuggingFaceViTEncoder()
    assert encoder.example_input_array.shape == (1, 3, 32, 32)
