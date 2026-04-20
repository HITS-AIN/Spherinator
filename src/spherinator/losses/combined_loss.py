import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Combines multiple loss functions as a weighted sum."""

    def __init__(self, losses: list[nn.Module], factors: list[float]):
        if len(losses) != len(factors):
            raise ValueError("losses and factors must have the same length")
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.factors = factors

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum(f * loss(x, target) for f, loss in zip(self.factors, self.losses))
