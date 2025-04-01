import math

import torch
import torch.nn as nn


class ConvolutionalDecoder1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        """ConvolutionalDecoder1D initializer
        Args:
            input_dim (int): The number of input features
            output_dim (int): The number of output features
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.example_input_array = torch.randn(1, 1, input_dim)

        dec2_input_dim = math.ceil(output_dim / 2)
        dec1_input_dim = math.ceil(dec2_input_dim / 2)

        self.dec1 = nn.Sequential(
            nn.LazyLinear(64 * dec1_input_dim),
            nn.Unflatten(1, (64, dec1_input_dim)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x
