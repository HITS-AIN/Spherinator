import torch
from hyperspherical_vae.distributions import VonMisesFisher
from torch.distributions.normal import Normal
from torch.distributions.von_mises import VonMises


def test_von_mises_fisher_1d():
    dist = VonMisesFisher(torch.Tensor([0.0]), torch.Tensor([1.0]))
    try:
        dist.sample()
        assert False
    except ValueError:
        assert True


def test_von_mises_fisher_2d():
    dist = VonMisesFisher(torch.Tensor([0.0, 0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == True
    assert torch.isclose(
        dist.sample(), torch.Tensor([-0.9916, 0.1294]), rtol=1e-3
    ).all()
    assert torch.isclose(
        dist.sample(), torch.Tensor([-0.7871, 0.6168]), rtol=1e-3
    ).all()


def test_von_mises_fisher_pytorch_1d():
    dist = VonMises(torch.Tensor([0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == False
    assert torch.isclose(dist.sample(), torch.Tensor([-0.8953]), rtol=1e-3)


def test_von_mises_fisher_pytorch_2d():
    dist = VonMises(torch.Tensor([0.0, 0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == False
    assert torch.isclose(
        dist.sample(), torch.Tensor([-0.8953, 1.8114]), rtol=1e-3
    ).all()


def test_normal_pytorch():
    dist = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == True
    assert torch.isclose(dist.sample(), torch.Tensor([1.5410]), rtol=1e-3)
