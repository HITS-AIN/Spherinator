import math

import torch
import torch.nn as nn


class ConvolutionalEncoder1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """ConvolutionalEncoder1D initializer
        Input shape: (batch_size, 1, input_dim)
        Output shape: (batch_size, output_dim)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.example_input_array = torch.randn(2, 1, input_dim)

        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        enc1_output_dim = math.floor(input_dim / 2)
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        enc2_output_dim = math.floor(enc1_output_dim / 2)
        self.enc3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(64 * enc2_output_dim), output_dim),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        return x
