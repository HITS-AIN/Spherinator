import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_sizes: list[int],
        output_size: int,
    ) -> None:
        super().__init__()

        self.latent_size = latent_size
        self.output_size = output_size

        self.example_input_array = torch.randn(1, latent_size)

        layers: list[nn.Module] = []
        prev = latent_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
