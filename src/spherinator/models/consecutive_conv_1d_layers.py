from typing import Callable, Optional

from torch import nn


class ConsecutiveConv1DLayer:
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        num_layers: int = 5,
        base_channel_number: int = 16,
        channel_increment: int = 4,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
        pooling: Optional[Callable[..., nn.Module]] = None,
        transpose: bool = False,
    ) -> None:
        """A class that defines a consecutive convolutional layer.
        Args:
            kernel_size (int, optional): The kernel size. Defaults to 3.
            stride (int, optional): The stride. Defaults to 1.
            padding (int, optional): The padding. Defaults to 0.
            num_layers (int, optional): The number of layers. Defaults to 5.
            base_channel_number (int, optional): The base channel number. Defaults to 16.
            channel_increment (int, optional): The channel increment. Defaults to 4.
            activation (Optional[Callable[..., nn.Module]], optional): The activation function.
            Defaults to nn.ReLU.
            norm (Optional[Callable[..., nn.Module]], optional): The normalization layer.
            Defaults to nn.BatchNorm1d.
            pooling (Optional[Callable[..., nn.Module]], optional): The pooling layer.
            Defaults to None.
            transpose (bool, optional): If the convolutional layer is a transpose convolutional layer.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_layers = num_layers
        self.base_channel_number = base_channel_number
        self.channel_increment = channel_increment
        self.activation = activation
        self.norm = norm
        self.pooling = pooling
        self.transpose = transpose

        if transpose:
            self.conv = nn.LazyConvTranspose1d
        else:
            self.conv = nn.LazyConv1d

    def __get_single_layer(self, out_channels: int) -> nn.Module:
        layers = []
        layers.append(
            self.conv(
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )
        if self.norm:
            layers.append(self.norm(out_channels))
        if self.activation:
            layers.append(self.activation())
        if self.pooling:
            layers.append(self.pooling())
        return nn.Sequential(*layers)

    def get_model(self) -> nn.Module:
        channel_numbers = [
            self.base_channel_number + i * self.channel_increment
            for i in range(0, self.num_layers)
        ]

        if self.transpose:
            # reverse the channel numbers
            channel_numbers = channel_numbers[::-1]

        layers = []
        for nb_channels in channel_numbers:
            layers.append(self.__get_single_layer(nb_channels))
        return nn.Sequential(*layers)
