import os
import sys
import torch
from torch.distributions.von_mises import VonMises
from torch.distributions.normal import Normal

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../external/s-vae-pytorch/'))
from hyperspherical_vae.distributions import (VonMisesFisher)


def test_von_mises_fisher_1d():
    torch.manual_seed(0)
    dist = VonMisesFisher(torch.Tensor([0.0]), torch.Tensor([1.0]))
    try:
        dist.sample()
        assert False
    except ValueError:
        assert True


def test_von_mises_fisher_2d():
    torch.manual_seed(0)
    dist = VonMisesFisher(torch.Tensor([0.0, 0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == True
    assert torch.isclose(dist.sample(), torch.Tensor([-0.9916,  0.1294]), rtol = 1e-3).all()
    assert torch.isclose(dist.sample(), torch.Tensor([-0.7871,  0.6168]), rtol = 1e-3).all()


def test_von_mises_fisher_pytorch():
    torch.manual_seed(0)
    dist = VonMises(torch.Tensor([0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == False
    assert torch.isclose(dist.sample(), torch.Tensor([-0.8953]), rtol = 1e-3)


def test_normal_pytorch():
    torch.manual_seed(0)
    dist = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

    assert dist.has_rsample == True
    assert torch.isclose(dist.sample(), torch.Tensor([1.5410]), rtol = 1e-3)
