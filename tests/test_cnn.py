import torch

from spherinator.models import (
    ConvolutionalDecoder256,
    ConvolutionalEncoder,
    RotationalVariationalAutoencoderPower,
)


def test_convolutional_encoder():
    """Check if weights are reproducible"""
    model = ConvolutionalEncoder(latent_dim=3)

    assert model.conv0.weight.shape == torch.Size([16, 3, 3, 3])
    assert torch.isclose(
        model.conv0.weight[0, 0, 0, 0], torch.Tensor([-0.0014]), atol=1e-3
    ).all()


def test_model():
    """Check if weights are reproducible"""
    model = RotationalVariationalAutoencoderPower(
        encoder=ConvolutionalEncoder(latent_dim=3)
    )

    assert model.encoder.conv0.weight.shape == torch.Size([16, 3, 3, 3])
    assert torch.isclose(
        model.encoder.conv0.weight[0, 0, 0, 0], torch.Tensor([-0.0014]), atol=1e-3
    ).all()


def test_convolutional_decoder():
    model = ConvolutionalDecoder256(latent_dim=3)
    data = torch.randn([2, 3])

    output = model(data)

    assert output.shape == torch.Size([2, 3, 256, 256])
