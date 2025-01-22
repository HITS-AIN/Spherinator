import pytest
import torch

from spherinator.models import ConvolutionalDecoder1D, ConvolutionalEncoder1D


@pytest.mark.parametrize(("input", "output"), [(128, 64), (343, 171), (344, 172)])
def test_Conv1d(input, output):
    data = torch.randn([2, 1, input])
    conv = torch.nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1)
    out = conv(data)

    assert out.shape == torch.Size([2, 32, output])


@pytest.mark.parametrize("input_dim", [128, 343, 344])
def test_convolutional_encoder_1d(input_dim):
    encoder = ConvolutionalEncoder1D(input_dim=input_dim, output_dim=3)
    data = torch.randn([2, 1, input_dim])
    out = encoder(data)

    assert out.shape == torch.Size([2, 3])


@pytest.mark.parametrize("output_dim", [128, 343, 344])
def test_convolutional_decoder_1d(output_dim):
    encoder = ConvolutionalDecoder1D(input_dim=3, output_dim=output_dim)
    data = torch.randn([2, 3])
    out = encoder(data)

    assert out.shape == torch.Size([2, 1, output_dim])
