import GMR_Conv
import pytest
import torch
import torchvision.models

from spherinator.models import ConvolutionalDecoder2D, GMRResNetDecoder, GMRResNetSpatialEncoder, VariationalAutoencoder


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
        (
            GMRResNetSpatialEncoder(
                block=GMR_Conv.GMRBasicBlock,
                layers=[2, 2, 2, 2],
                input_dim=[3, 128, 128],
                layer_stride=[2, 2, 2, 2],
                latent_channels=3,
            ),
            GMRResNetDecoder(
                input_channels=3,
                output_dim=[3, 128, 128],
                layer_stride=[2, 2, 2, 2],
            ),
            (3, 128, 128),
            48,
            [3, 4, 4],
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
    input = torch.randn(2, *input_dim)
    batch_size = input.shape[0]

    (z_mean, z_var), (_, _), _, recon = model(input)

    assert z_mean.shape == (batch_size, z_dim)
    assert z_var.shape == (batch_size, 1)
    assert recon.shape == input.shape
