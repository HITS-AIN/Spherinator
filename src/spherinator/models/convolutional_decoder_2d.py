from typing import Optional

import torch
import torch.nn as nn

from .consecutive_conv_transpose_2d_layers import ConsecutiveConvTranspose2DLayer
from .weights_provider import WeightsProvider


class ConvolutionalDecoder2D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: list[int],
        cnn_input_dim: list[int],
        cnn_layers: list[ConsecutiveConvTranspose2DLayer],
        weights: Optional[WeightsProvider] = None,
        freeze: bool = False,
    ) -> None:
        """ConvolutionalDecoder1DGen initializer
        Args:
            input_dim (int): The number of input features
            output_dim (list[int]): The number of output features
            cnn_input_dim (list[int]): The number of input features
            cnn_layers (list[ConsecutiveConvTranspose2DLayer]): The list of consecutive convolutional layers
            weights (Optional[WeightsProvider], optional): The weights to load. Defaults to None.
            freeze (bool, optional): Whether to freeze the model. Defaults to False.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cnn_layers = cnn_layers

        self.example_input_array = torch.randn(1, input_dim)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, cnn_input_dim[0] * cnn_input_dim[1]),
            nn.Unflatten(1, cnn_input_dim),
            nn.BatchNorm1d(cnn_input_dim[0]),
            nn.ReLU(),
        )

        layers = []
        for layer in cnn_layers:
            layers.append(layer.get_model())
        self.cnn = nn.Sequential(*layers)

        if weights is not None:
            self.load_state_dict(weights.get_state_dict())
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.cnn(x)
        return x
