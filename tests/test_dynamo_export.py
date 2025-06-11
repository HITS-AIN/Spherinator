import pytest
import torch
import torch.nn as nn
from power_spherical import PowerSpherical


class DistributionModel(torch.nn.Module):
    def __init__(self, dist):
        self.dist = dist
        super().__init__()

    def forward(self, x):
        return self.dist.rsample(x.shape)


@pytest.mark.parametrize(
    ("module", "input"),
    [
        (nn.Linear(3, 2), torch.randn(2, 3)),
        (DistributionModel(torch.distributions.normal.Normal(0, 1)), torch.randn(2, 3)),
        pytest.param(
            DistributionModel(PowerSpherical(torch.Tensor([0.0, 1.0]), torch.Tensor([1.0]))),
            torch.randn(2, 3),
            marks=pytest.mark.xfail(
                reason="torch._dynamo.exc.UserError: Tried to use data-dependent value in the subsequent computation."
            ),
        ),
    ],
)
def test_dynamo_export(module, input):
    module(input)
    torch.export.export(module, args=(input,))
