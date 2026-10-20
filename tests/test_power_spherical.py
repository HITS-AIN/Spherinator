import torch
from power_spherical import HypersphericalUniform, PowerSpherical


def test_power_spherical_2d():
    dist = PowerSpherical(torch.Tensor([0.0, 0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample is True
    samples = dist.rsample((16,))
    assert samples.shape == (16, 1, 2)
    assert torch.allclose(samples.norm(dim=-1), torch.ones(16, 1), atol=1e-5)


def test_power_spherical_large_scale():
    z_mean = torch.Tensor([1.0, 0.0, 0.0])
    z_mean = torch.nn.functional.normalize(z_mean, p=2.0, dim=0)
    z_scale = torch.Tensor([1.0e38])
    z_scale = torch.nn.functional.softplus(z_scale) + 1
    dist = PowerSpherical(z_mean, z_scale)

    samples = dist.rsample((16,))
    assert torch.isfinite(samples).all()
    assert torch.allclose(samples.norm(dim=-1), torch.ones(16, 1), atol=1e-5)
    assert samples[..., 0].mean() > 0.99


def test_power_spherical_2d_batch():
    batch_size = 32
    loc = torch.randn(batch_size, 3)
    scale = torch.ones(batch_size)
    dist = PowerSpherical(loc, scale)

    sample = dist.rsample()
    assert sample.shape == torch.Size([batch_size, 3])


def test_kl_divergence():
    dim = 8
    loc = torch.tensor([0.0] * (dim - 1) + [1.0])
    scale = torch.tensor(10.0)

    dist1 = PowerSpherical(loc, scale)
    dist2 = HypersphericalUniform(dim)
    x = dist1.sample((100000,))

    assert torch.allclose(
        (dist1.log_prob(x) - dist2.log_prob(x)).mean(),
        torch.distributions.kl_divergence(dist1, dist2),
        atol=1e-2,
    )
