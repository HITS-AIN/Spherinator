import torch
import torch.nn as nn

from .consecutive_conv_transpose_1d_layers import ConsecutiveConvTranspose1DLayer


class ConvolutionalDecoder1DGen(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: list[int],
        cnn_input_dim: list[int],
        cnn_layers: list[ConsecutiveConvTranspose1DLayer],
    ) -> None:
        """ConvolutionalDecoder1DGen initializer
        Args:
            input_dim (int): The number of input features
            output_dim (int): The number of output features
            cnn_layers (list[ConsecutiveConvTranspose1DLayer]): The list of consecutive convolutional layers
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.cnn(x)
        return x
