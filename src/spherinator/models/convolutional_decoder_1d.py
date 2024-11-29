import torch
import torch.nn as nn


class ConvolutionalDecoder1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """ConvolutionalDecoder1D initializer
        Input shape: (batch_size, input_dim)
        Output shape: (batch_size, 1, output_dim)
        """
        super().__init__()

        self.dec1 = nn.Sequential(
            nn.Linear(input_dim, int(64 * output_dim / 4)),
            nn.Unflatten(1, (64, int(output_dim / 4))),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )  # 64 x (output_dim / 4)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )  # 32 x (output_dim / 2)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1),
        )  # output_dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x
