import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        latent_size: int,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.latent_size = latent_size

        self.example_input_array = torch.randn(1, input_size)

        layers: list[nn.Module] = [nn.Flatten()]
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, latent_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
