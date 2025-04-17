from typing import Callable, Optional

from torch import nn


class ConsecutiveConv1DLayer:
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        out_channels: list[int] = [1],
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
        pooling: Optional[nn.Module] = None,
    ) -> None:
        """A class that defines a consecutive convolutional layer.
        Args:
            kernel_size (int, optional): The kernel size. Defaults to 3.
            stride (int, optional): The stride. Defaults to 1.
            padding (int, optional): The padding. Defaults to 0.
            out_channels (list[int], optional): The list of output channels. Defaults to [1].
            activation (Optional[Callable[..., nn.Module]], optional): The activation function.
            Defaults to nn.ReLU.
            norm (Optional[Callable[..., nn.Module]], optional): The normalization layer.
            Defaults to nn.BatchNorm1d.
            pooling (Optional[Callable[..., nn.Module]], optional): The pooling layer.
            Defaults to None.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.activation = activation
        self.norm = norm
        self.pooling = pooling

    def __get_single_layer(self, out_channels: int) -> nn.Module:
        layers = []
        layers.append(
            nn.LazyConv1d(
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
            layers.append(self.pooling)
        return nn.Sequential(*layers)

    def get_model(self) -> nn.Module:
        layers = []
        for out_channels in self.out_channels:
            layers.append(self.__get_single_layer(out_channels))
        return nn.Sequential(*layers)
