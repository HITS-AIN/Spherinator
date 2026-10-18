import pytest
import torch

from spherinator.distributions import truncated_normal_distribution


def test_density_uses_natural_exponential_and_zero_outside_support():
    x = torch.tensor([0.0, 2.0])
    mu = torch.zeros(2)
    sigma = torch.ones(2)

    density = truncated_normal_distribution(x, mu, sigma, -1.0, 1.0)
    expected_at_zero = torch.tensor(0.39894228) / (
        torch.distributions.Normal(0, 1).cdf(torch.tensor(1.0))
        - torch.distributions.Normal(0, 1).cdf(torch.tensor(-1.0))
    )

    assert torch.isclose(density[0], expected_at_zero)
    assert density[1] == 0


def test_all_standard_deviations_must_be_positive():
    with pytest.raises(ValueError, match="Every standard deviation"):
        truncated_normal_distribution(torch.zeros(2), torch.zeros(2), torch.tensor([1.0, 0.0]), -1.0, 1.0)
