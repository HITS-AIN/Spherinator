import torch
import torch.nn as nn


class DenseModel(nn.Module):
    def __init__(self, layer_dims: list[int]):
        """DenseModel initializer"""
        super().__init__()

        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]

        self.example_input_array = torch.randn(2, self.input_dim)

        modules = []
        num_layers = len(layer_dims)
        for i in range(num_layers - 2):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            modules.append(nn.ReLU())
        modules.append(
            nn.Linear(layer_dims[num_layers - 2], layer_dims[num_layers - 1])
        )

        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.model(x)
        return x
