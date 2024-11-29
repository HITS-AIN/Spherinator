import pytest
import torch

from spherinator.models import ConvolutionalDecoder1D, ConvolutionalEncoder1D


@pytest.mark.parametrize(("input", "output"), [(128, 64), (343, 171), (344, 172)])
def test_Conv1d(input, output):
    data = torch.randn([2, 1, input])
    conv = torch.nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1)
    out = conv(data)

    assert out.shape == torch.Size([2, 32, output])


def test_convolutional_encoder_1d():
    encoder = ConvolutionalEncoder1D(input_dim=128, output_dim=3)
    data = torch.randn([2, 1, 128])
    out = encoder(data)

    assert out.shape == torch.Size([2, 3])


def test_convolutional_decoder_1d():
    encoder = ConvolutionalDecoder1D(input_dim=3, output_dim=128)
    data = torch.randn([2, 3])
    out = encoder(data)

    assert out.shape == torch.Size([2, 1, 128])


def test_convolutional_encoder_1d_344():
    encoder = ConvolutionalEncoder1D(input_dim=344, output_dim=3)
    data = torch.randn([2, 1, 344])
    out = encoder(data)

    assert out.shape == torch.Size([2, 3])
