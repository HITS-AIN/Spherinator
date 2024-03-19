import pytest
import torchvision.models

import models


@pytest.mark.parametrize(
    "encoder, decoder, input_size",
    [
        (
            torchvision.models.resnet18(num_classes=256),
            models.ConvolutionalDecoder(),
            128,
        ),
        (
            torchvision.models.vit_b_16(num_classes=256),
            models.ConvolutionalDecoder224(),
            224,
        ),
    ],
)
def test_resnet(encoder, decoder, input_size):
    z_dim = 2
    h_dim = 256
    model = models.RotationalVariationalAutoencoderPower(
        z_dim=z_dim,
        h_dim=h_dim,
        input_size=input_size,
        encoder=encoder,
        decoder=decoder,
    )
    input = model.example_input_array
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
