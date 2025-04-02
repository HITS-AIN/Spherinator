import pytest
import torch
import torchvision.models

from spherinator.models import ConvolutionalDecoder2D, VariationalAutoencoder


@pytest.mark.parametrize(
    "encoder, decoder, input_dim",
    [
        (
            torchvision.models.resnet18(num_classes=10),
            ConvolutionalDecoder2D(3, [3, 128, 128], [3, 128, 128]),
            (3, 128, 128),
        ),
        (
            torchvision.models.vit_b_16(num_classes=10),
            ConvolutionalDecoder2D(3, [3, 224, 224], [3, 224, 224]),
            (3, 224, 224),
        ),
    ],
)
def test_resnet(encoder, decoder, input_dim):
    z_dim = 3
    model = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=10,
        z_dim=z_dim,
    )
    input = torch.randn(2, *input_dim)
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
