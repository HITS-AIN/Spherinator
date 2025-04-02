import pytest
import torchvision.models

from spherinator.models import (
    ConsecutiveConv2DLayer,
    ConvolutionalDecoder2D,
    VariationalAutoencoder,
)


@pytest.mark.parametrize(
    "encoder, decoder, input_size",
    [
        (
            torchvision.models.resnet18(num_classes=256),
            ConvolutionalDecoder2D(
                input_dim=256,
                output_dim=[128, 128],
                cnn_input_dim=[128, 128],
            ),
            128,
        ),
        (
            torchvision.models.vit_b_16(num_classes=256),
            ConvolutionalDecoder2D(
                input_dim=256,
                output_dim=[128, 128],
                cnn_input_dim=[128, 128],
            ),
            224,
        ),
    ],
)
def test_resnet(encoder, decoder, input_size):
    z_dim = 2
    h_dim = 256
    model = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        encoder_out_dim=h_dim,
        z_dim=z_dim,
    )
    input = model.example_input_array
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
