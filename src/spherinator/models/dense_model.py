from typing import Callable, Optional

import torch
import torch.nn as nn


class DenseModel(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        output_shape: Optional[list[int]] = None,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ):
        """DenseModel initializer
        Args:
            layer_dims (list[int]): The list of layer dimensions
            output_shape (Optional[list[int]], optional): The output shape. Defaults to None.
        """
        super().__init__()

        self.example_input_array = torch.randn(1, layer_dims[0])

        layers = []
        layers.append(nn.Flatten())
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2 and activation:
                layers.append(activation())

        if output_shape:
            layers.append(nn.Unflatten(1, output_shape))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
