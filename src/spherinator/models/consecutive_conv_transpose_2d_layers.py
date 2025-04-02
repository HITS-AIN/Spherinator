from typing import Callable, Optional

from torch import nn


class ConsecutiveConvTranspose2DLayer:
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        out_channels_list: list[int] = [1],
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        pooling: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """A class that defines a consecutive convolutional layer.
        Args:
            kernel_size (int, optional): The kernel size. Defaults to 3.
            stride (int, optional): The stride. Defaults to 1.
            padding (int, optional): The padding. Defaults to 0.
            out_channels_list (list[int], optional): The list of output channels.
            activation (Optional[Callable[..., nn.Module]], optional): The activation function.
            Defaults to nn.ReLU.
            norm (Optional[Callable[..., nn.Module]], optional): The normalization layer.
            Defaults to nn.BatchNorm2d.
            pooling (Optional[Callable[..., nn.Module]], optional): The pooling layer.
            Defaults to None.
            transpose (bool, optional): If the convolutional layer is a transpose convolutional layer.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels_list = out_channels_list
        self.activation = activation
        self.norm = norm
        self.pooling = pooling

    def __get_single_layer(self, out_channels: int) -> nn.Module:
        layers = []
        layers.append(
            nn.LazyConvTranspose2d(
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
        layers = []
        for out_channels in self.out_channels_list:
            layers.append(self.__get_single_layer(out_channels))
        return nn.Sequential(*layers)
