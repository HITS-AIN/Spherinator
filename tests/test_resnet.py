import torchvision.models

import models


def test_resnet():
    z_dim = 2
    h_dim = 256
    model = models.RotationalVariationalAutoencoderPower(
        z_dim=z_dim,
        h_dim=h_dim,
        encoder=torchvision.models.resnet18(num_classes=h_dim),
    )
    input = model.example_input_array
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
