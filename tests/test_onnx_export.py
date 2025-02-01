import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import PowerSpherical

from spherinator.models import (
    Autoencoder,
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
)


class Model1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        return out


class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DistributionModel(torch.nn.Module):
    def __init__(self, dist):
        self.dist = dist
        super().__init__()

    def forward(self, x):
        return self.dist.rsample(x.shape)


@pytest.mark.parametrize(
    ("module", "input"),
    [
        (Model1(), torch.randn(2, 2, 2)),
        (Model2(), torch.randn(1, 1, 32, 32)),
        pytest.param(
            DistributionModel(
                PowerSpherical(torch.Tensor([0.0, 1.0]), torch.Tensor([1.0]))
            ),
            torch.randn(2, 3),
            marks=pytest.mark.xfail,
        ),
        (
            ConvolutionalEncoder1D(12, 24),
            torch.randn(2, 1, 12),
        ),
        pytest.param(
            Autoencoder(
                ConvolutionalEncoder1D(12, 24), ConvolutionalDecoder1D(24, 12), 24, 3
            ),
            torch.randn(2, 1, 12),
            marks=pytest.mark.xfail(
                sys.version_info.minor == 9, reason="Fails on Python 3.9"
            ),
        ),
    ],
)
def test_onnx_dynamo_export(module, input):
    module.eval()
    module(input)
    torch.onnx.dynamo_export(module, input)
