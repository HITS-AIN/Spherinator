import torch
from torch import nn

from spherinator.models import (
    ConsecutiveConv1DLayer,
    ConsecutiveConvTranspose1DLayer,
    ConvolutionalDecoder1D,
    ConvolutionalEncoder1D,
)


def test_convolutional_encoder_1d():
    input_dim = [3, 12]
    encoder = ConvolutionalEncoder1D(
        input_dim=input_dim,
        output_dim=3,
        cnn_layers=[ConsecutiveConv1DLayer(out_channels=[6, 3])],
    )
    data = torch.randn([2, *input_dim])
    out = encoder(data)

    assert out.shape == torch.Size([2, 3])


def test_consecutive_conv_1d_layer():
    conv = ConsecutiveConv1DLayer(
        out_channels=[3],
        pooling=nn.MaxPool1d(2),
    )
    model = conv.get_model()

    print(model)

    data = torch.randn([1, 1, 32])
    out = model(data)

    assert out.shape == torch.Size([1, 3, 15])


def test_convolutional_decoder_1d():
    output_dim = [1, 128]
    cnn_layers = [ConsecutiveConvTranspose1DLayer(out_channels=[16, output_dim[0]])]
    decoder = ConvolutionalDecoder1D(
        input_dim=3,
        output_dim=output_dim,
        cnn_input_dim=[20, 124],
        cnn_layers=cnn_layers,
    )
    data = torch.randn([2, 3])
    out = decoder(data)

    assert out.shape == torch.Size([2, *output_dim])


def test_convolutional_decoder_1d_deep():
    cnn_layers = [
        ConsecutiveConvTranspose1DLayer(
            kernel_size=5,
            out_channels=list(range(248, 12, -4)) + [1],
        )
    ]
    decoder = ConvolutionalDecoder1D(
        input_dim=3,
        output_dim=[1, 344],
        cnn_input_dim=[252, 104],
        cnn_layers=cnn_layers,
    )
    data = torch.randn([1, 3])
    out = decoder(data)

    assert out.shape == torch.Size([1, 1, 344])
