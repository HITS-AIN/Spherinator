from typing import Callable, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple multi-layer perceptron (MLP) model."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        """
        Args:
            input_size (int): The size of the input features.
            hidden_sizes (list[int]): A list of hidden layer sizes.
            output_size (int): The size of the output features.
            activation (Optional[Callable[..., nn.Module]]): The activation function to use. Defaults to nn.ReLU.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.example_input_array = torch.randn(1, input_size)

        layers: list[nn.Module] = [nn.Flatten()]
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
