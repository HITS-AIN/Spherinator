import pytest
import torch
import torchvision.models

from spherinator.models import ConvolutionalDecoder2D, UpsamplingDecoder2D, VariationalAutoencoder


def test_upsampling_decoder_supports_non_square_output():
    decoder = UpsamplingDecoder2D(input_dim=3, output_dim=[2, 13, 17], base_channels=16, seed_size=4)

    output = decoder(torch.randn(2, 3))

    assert output.shape == (2, 2, 13, 17)


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
