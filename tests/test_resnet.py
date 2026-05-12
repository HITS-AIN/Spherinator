import pytest
import torch
import torchvision.models

from spherinator.models import (
    ConvolutionalDecoder2D,
    VariationalAutoencoder,
)


@pytest.mark.parametrize(
    "encoder, decoder, input_dim, z_dim, encoder_out_dim",
    [
        (
            torchvision.models.resnet18(num_classes=10),
            ConvolutionalDecoder2D(3, [3, 128, 128], [3, 128, 128]),
            (3, 128, 128),
            3,
            10,
        ),
        (
            torchvision.models.vit_b_16(num_classes=10),
            ConvolutionalDecoder2D(3, [3, 224, 224], [3, 224, 224]),
            (3, 224, 224),
            3,
            10,
        ),
    ],
)
def test_resnet(encoder, decoder, input_dim, z_dim, encoder_out_dim):
    model = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=encoder_out_dim,
        z_dim=z_dim,
    )
    bs = 2
    x = torch.randn(bs, *input_dim)

    (z_mean, z_var), (_, _), _, recon = model(x)

    assert z_mean.shape == (bs, z_dim)
    assert z_var.shape == (bs, 1)
    assert recon.shape == x.shape
