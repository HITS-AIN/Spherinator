import pytest
import torch
from torch import nn

import spherinator.losses.perceptional_loss as perceptual_module


class Capture(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, value):
        self.value = value
        return value


class TinyVGG:
    def __init__(self, capture):
        self.features = nn.Sequential(capture)


def test_perceptual_loss_applies_imagenet_normalization(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(perceptual_module, "vgg16", lambda **_kwargs: TinyVGG(capture))
    loss = perceptual_module.PerceptualLoss(layers=(0,), weights=(1.0,))
    image = torch.ones(1, 1, 4, 4)

    assert loss(image, image) == 0
    expected = (torch.ones(1, 3, 4, 4) - loss.mean) / loss.std
    assert torch.allclose(capture.value, expected)


def test_perceptual_loss_normalization_not_in_state_dict(monkeypatch):
    monkeypatch.setattr(perceptual_module, "vgg16", lambda **_kwargs: TinyVGG(Capture()))
    loss = perceptual_module.PerceptualLoss(layers=(0,), weights=(1.0,))

    assert "mean" not in loss.state_dict()
    assert "std" not in loss.state_dict()


def test_perceptual_loss_validates_configuration(monkeypatch):
    monkeypatch.setattr(perceptual_module, "vgg16", lambda **_kwargs: TinyVGG(Capture()))

    with pytest.raises(ValueError, match="same length"):
        perceptual_module.PerceptualLoss(layers=(0,), weights=(1.0, 2.0))
