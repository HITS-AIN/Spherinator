import torch
from torch import nn

from spherinator.models import (
    ConsecutiveConv2DLayer,
    ConsecutiveConvTranspose2DLayer,
    ConvolutionalDecoder2D,
    ConvolutionalEncoder2D,
)


def test_convolutional_encoder_2D():
    input_dim = [3, 128, 128]
    encoder = ConvolutionalEncoder2D(
        input_dim=input_dim,
        output_dim=3,
        cnn_layers=[ConsecutiveConv2DLayer(num_layers=2)],
    )
    input = torch.randn(2, *input_dim)
    out = encoder(input)

    assert out.shape == torch.Size([2, 3])


def test_consecutive_conv_2D_layer():
    conv = ConsecutiveConv2DLayer(
        num_layers=1,
        base_channel_number=3,
        pooling=nn.MaxPool2d(2),
    )
    model = conv.get_model()

    print(model)

    data = torch.randn([2, 3, 32, 32])
    out = model(data)

    assert out.shape == torch.Size([2, 3, 15, 15])


def test_convolutional_decoder_2D():
    output_dim = [1, 128, 128]
    decoder = ConvolutionalDecoder2D(
        input_dim=3,
        output_dim=output_dim,
        cnn_input_dim=[20, 124, 124],
        cnn_layers=[
            ConsecutiveConvTranspose2DLayer(
                out_channels_list=[16, output_dim[0]], norm=None
            )
        ],
    )
    data = torch.randn([2, 3])
    out = decoder(data)

    assert out.shape == torch.Size([2, *output_dim])


def test_convolutional_decoder_2D_deep():
    out_channels_list = list(range(248, 12, -4)) + [1]

    assert len(out_channels_list) == 60

    cnn_layers = [
        ConsecutiveConvTranspose2DLayer(
            kernel_size=5,
            out_channels_list=out_channels_list,
        )
    ]
    decoder = ConvolutionalDecoder2D(
        input_dim=3,
        output_dim=[1, 344, 344],
        cnn_input_dim=[252, 104, 104],
        cnn_layers=cnn_layers,
    )
    data = torch.randn([1, 3])
    out = decoder(data)

    assert out.shape == torch.Size([1, 1, 344, 344])
