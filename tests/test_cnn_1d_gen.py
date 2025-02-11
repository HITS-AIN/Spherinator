import pytest
import torch

from spherinator.models import (
    ConsecutiveConv1DLayer,
    ConsecutiveConvTranspose1DLayer,
    ConvolutionalDecoder1DGen,
    ConvolutionalEncoder1DGen,
)


@pytest.mark.parametrize("input_dim", [(1, 128)])
def test_convolutional_encoder_1d_gen(input_dim):
    cnn_layers = [
        ConsecutiveConv1DLayer(
            num_layers=2,
        )
    ]
    encoder = ConvolutionalEncoder1DGen(
        input_dim=input_dim,
        output_dim=3,
        cnn_layers=cnn_layers,
    )
    data = torch.randn([2, *input_dim])
    out = encoder(data)

    assert out.shape == torch.Size([2, 3])


@pytest.mark.parametrize("output_dim", [(1, 128)])
def test_convolutional_decoder_1d_gen(output_dim):
    cnn_layers = [
        ConsecutiveConvTranspose1DLayer(out_channels_list=[16, output_dim[0]])
    ]
    decoder = ConvolutionalDecoder1DGen(
        input_dim=3,
        output_dim=output_dim,
        cnn_input_dim=(20, 124),
        cnn_layers=cnn_layers,
    )
    data = torch.randn([2, 3])
    out = decoder(data)

    assert out.shape == torch.Size([2, *output_dim])


def test_convolutional_decoder_1d_gen_deep():
    out_channels_list = list(range(248, 12, -4)) + [1]

    assert len(out_channels_list) == 60

    cnn_layers = [
        ConsecutiveConvTranspose1DLayer(
            kernel_size=5,
            out_channels_list=out_channels_list,
        )
    ]
    decoder = ConvolutionalDecoder1DGen(
        input_dim=3,
        output_dim=(1, 344),
        cnn_input_dim=(252, 104),
        cnn_layers=cnn_layers,
    )
    data = torch.randn([1, 3])
    out = decoder(data)

    assert out.shape == torch.Size([1, 1, 344])
