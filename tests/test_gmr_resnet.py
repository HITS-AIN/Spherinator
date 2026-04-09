import torch

from spherinator.models.gmr_resnet_decoder import GMRResNetDecoder
from spherinator.models.gmr_resnet_spatial_encoder import GMRResNetSpatialEncoder


def test_gmr_resnet_spatial_encoder_forward():
    from GMR_Conv import GMRBasicBlock

    input_dim = [3, 128, 128]
    latent_channels = 64

    encoder = GMRResNetSpatialEncoder(
        block=GMRBasicBlock,
        layers=[2, 2, 2, 2],
        input_dim=input_dim,
        latent_channels=latent_channels,
    )
    encoder.eval()

    x = torch.randn(2, *input_dim)
    with torch.no_grad():
        out = encoder(x)

    assert out.shape == (2, latent_channels, 8, 8)


def test_gmr_resnet_decoder_forward():
    input_channels = 64
    output_dim = [3, 128, 128]

    decoder = GMRResNetDecoder(
        input_channels=input_channels,
        output_dim=output_dim,
        layers=[2, 2, 2, 2],
    )
    decoder.eval()

    # spatial latent: (B, input_channels, seed_h, seed_w)
    # with default layer_stride=[1,2,2,2] and skip_first_maxpool=False:
    # seed = 128 // 2 // 1 // 2 // 2 // 2 = 8
    x = torch.randn(2, input_channels, 8, 8)
    with torch.no_grad():
        out = decoder(x)

    assert out.shape == (2, *output_dim)
    assert out.shape == (2, *output_dim)
    assert out.shape == (2, *output_dim)
