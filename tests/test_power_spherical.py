import os
import sys
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../external/power_spherical/'))
from power_spherical import PowerSpherical, HypersphericalUniform


def test_power_spherical_2d():
    torch.manual_seed(0)
    dist = PowerSpherical(torch.Tensor([0.0, 0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == True
    assert torch.isclose(dist.rsample(), torch.Tensor([-0.9971,  0.0766]), rtol = 1e-3).all()
    assert torch.isclose(dist.rsample(), torch.Tensor([-0.5954,  0.8034]), rtol = 1e-3).all()

def test_power_spherical_2d_batch():
    torch.manual_seed(0)
    batch_size = 32
    loc = torch.randn(batch_size, 3)
    scale = torch.ones(batch_size)
    dist = PowerSpherical(loc, scale)

    sample = dist.rsample()
    assert sample.shape == torch.Size([batch_size, 3])

def test_kl_divergence():
    dim = 8
    loc = torch.tensor([0.] * (dim - 1) + [1.])
    scale = torch.tensor(10.)

    dist1 = PowerSpherical(loc, scale)
    dist2 = HypersphericalUniform(dim)
    x = dist1.sample((100000,))

    assert torch.isclose(
        (dist1.log_prob(x) - dist2.log_prob(x)).mean(),
        torch.distributions.kl_divergence(dist1, dist2),
        atol=1e-2,
    ).all()
