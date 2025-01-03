import torch
import torch.nn as nn


class DenseEncoder(nn.Module):
    def __init__(self, layer_dims: list[int]):
        """ConvolutionalEncoder1D initializer"""
        super().__init__()

        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]

        self.example_input_array = torch.randn(2, self.input_dim)

        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*modules)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encoder(x)
        return x
