import GMR_Conv
import pytest
import torch
import torchvision.models

from spherinator.models import (
    ConvolutionalDecoder2D,
    GMRResNetSpatialEncoder,
    Sequential,
    UpsamplingDecoder2D,
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
        (
            Sequential(
                modules=[
                    GMRResNetSpatialEncoder(
                        block=GMR_Conv.GMRBasicBlock,
                        layers=[2, 2, 2, 2],
                        input_dim=[3, 128, 128],
                        layer_stride=[2, 2, 2, 2],
                        latent_channels=16,
                    ),
                    torch.nn.AdaptiveAvgPool2d(output_size=1),
                ]
            ),
            UpsamplingDecoder2D(
                input_dim=3,
                output_dim=[3, 128, 128],
                base_channels=256,
                seed_size=4,
            ),
            (3, 128, 128),
            3,
            16,
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
