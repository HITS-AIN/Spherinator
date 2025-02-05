from typing import Callable, Optional

import torch
import torch.nn as nn


class ConvolutionalEncoder1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        in_channels: int = 1,
        inc_channels: int = 3,
        kernel_size: int = 3,
        number_of_layers: int = 1,
        striding: int = 1,
        padding: int = 0,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
        pooling: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """ConvolutionalEncoder1D initializer
        Args:
            input_dim (int): The number of input features
            output_dim (int): The number of output features
            kernel_size (int, optional): The kernel size. Defaults to 5.
            number_of_layers (int, optional): The number of layers. Defaults to 3.
            striding (int, optional): The stride. Defaults to 1.
            padding (int, optional): The padding. Defaults to 0.
            activation (Optional[Callable[..., nn.Module]], optional): The activation function.
            Defaults to nn.ReLU.
            norm (Optional[Callable[..., nn.Module]], optional): The normalization layer.
            Defaults to nn.BatchNorm1d.
            pooling (Optional[Callable[..., nn.Module]], optional): The pooling layer.
            Defaults to None.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.number_of_layers = number_of_layers
        self.striding = striding
        self.padding = padding
        self.activation = activation
        self.norm = norm
        self.pooling = pooling

        self.example_input_array = torch.randn(1, 1, input_dim)

        layers = []
        nb_channels = in_channels
        for _ in range(number_of_layers):
            out_channels = nb_channels + inc_channels
            layers.append(
                nn.Conv1d(
                    nb_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=striding,
                    padding=padding,
                )
            )
            if norm is not None:
                layers.append(norm(out_channels))
            if activation is not None:
                layers.append(activation())
            if pooling is not None:
                layers.append(pooling())
            nb_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.fc(x)
        return x
