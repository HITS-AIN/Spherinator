import torch
import torch.nn as nn


class ConvolutionalEncoder1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        """ConvolutionalEncoder1D initializer
        Args:
            input_dim (int): The number of input features
            output_dim (int): The number of output features
        """
        super().__init__()

        assert input_dim % 4 == 0, "input_dim must be divisible by 4"

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.example_input_array = torch.randn(2, 1, input_dim)

        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )  # 32 x (input_dim / 2)
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )  # 64 x (input_dim / 4)
        self.enc3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(64 * input_dim / 4), output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        return x
