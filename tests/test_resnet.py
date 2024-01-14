import torch
import torchvision.models as models

from models import RotationalVariationalAutoencoderPower


def test_resnet():
    z_dim = 2
    model = RotationalVariationalAutoencoderPower(
        z_dim=z_dim,
        encoder=models.resnet18,
    )
    input = model.example_input_array
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
