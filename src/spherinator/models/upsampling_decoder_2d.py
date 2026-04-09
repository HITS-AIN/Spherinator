from typing import Optional

import torch
import torch.nn as nn

from .weights_provider import WeightsProvider


class _UpsampleBlock(nn.Module):
    """Bilinear upsample (2×) followed by a conv, BN, ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsamplingDecoder2D(nn.Module):
    """Image decoder using bilinear upsampling + convolutions.

    Avoids the checkerboard artifacts of transposed convolutions.
    Designed to decode a low-dimensional latent vector back to a
    full-resolution image matching the ViT input (224×224 by default).

    Architecture:
        z  →  Linear  →  reshape (base_channels, seed_size, seed_size)
           →  5× UpsampleBlock (each 2×)  →  final 1×1 conv  →  output
        Spatial path: seed_size × 2^n_upsample = output_size

    Args:
        input_dim (int): Latent vector size.
        output_dim (list[int]): Output shape [C, H, W].
        base_channels (int): Channels at the spatial seed. Defaults to 512.
        seed_size (int): Spatial size of the seed feature map. Defaults to 7.
        weights (Optional[WeightsProvider]): Optional pre-trained weights.
        freeze (bool): Freeze all parameters. Defaults to False.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: list[int],
        base_channels: int = 512,
        seed_size: int = 7,
        weights: Optional[WeightsProvider] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        out_channels, out_h, out_w = output_dim

        self.example_input_array = torch.randn(1, input_dim)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, base_channels * seed_size * seed_size),
            nn.Unflatten(1, (base_channels, seed_size, seed_size)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Build upsample blocks: halve channels each step
        blocks = []
        in_ch = base_channels
        current_size = seed_size
        while current_size < out_h // 2:
            out_ch = max(in_ch // 2, 32)
            blocks.append(_UpsampleBlock(in_ch, out_ch))
            in_ch = out_ch
            current_size *= 2
        # Final upsample to reach out_h
        blocks.append(_UpsampleBlock(in_ch, max(in_ch // 2, 32)))
        in_ch = max(in_ch // 2, 32)
        self.upsample_blocks = nn.Sequential(*blocks)

        # 1×1 projection to output channels + Sigmoid to keep values in [0,1]
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        if weights is not None:
            self.load_state_dict(weights.get_state_dict())
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.upsample_blocks(x)
        x = self.head(x)
        return x
