import pytest
import torch
from torch import nn

from spherinator.models import (
    ConsecutiveConv1DLayer,
    ConsecutiveConvTranspose1DLayer,
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
)


@pytest.mark.parametrize("input_dim", [(1, 128)])
def test_convolutional_encoder_1d_gen(input_dim):
    cnn_layers = [
        ConsecutiveConv1DLayer(
            num_layers=2,
        )
    ]
    encoder = ConvolutionalEncoder1D(
        input_dim=input_dim,
        output_dim=3,
        cnn_layers=cnn_layers,
    )
    data = torch.randn([2, *input_dim])
    out = encoder(data)

    assert out.shape == torch.Size([2, 3])


def test_consecutive_conv_1d_layer():
    conv = ConsecutiveConv1DLayer(
        num_layers=1,
        base_channel_number=3,
        pooling=nn.MaxPool1d(2),
    )
    model = conv.get_model()

    print(model)

    data = torch.randn([1, 1, 32])
    out = model(data)

    assert out.shape == torch.Size([1, 3, 15])


@pytest.mark.parametrize("output_dim", [(1, 128)])
def test_convolutional_decoder_1d_gen(output_dim):
    cnn_layers = [
        ConsecutiveConvTranspose1DLayer(out_channels_list=[16, output_dim[0]])
    ]
    decoder = ConvolutionalDecoder1D(
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
    decoder = ConvolutionalDecoder1D(
        input_dim=3,
        output_dim=(1, 344),
        cnn_input_dim=(252, 104),
        cnn_layers=cnn_layers,
    )
    data = torch.randn([1, 3])
    out = decoder(data)

    assert out.shape == torch.Size([1, 1, 344])
