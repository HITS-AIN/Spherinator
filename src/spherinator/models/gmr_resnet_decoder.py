from typing import Optional

import torch
import torch.nn as nn

from .weights_provider import WeightsProvider


class _DecoderBasicBlock(nn.Module):
    """Residual block for decoder, mirroring GMRBasicBlock.

    Optionally upsamples spatially (reverse of encoder's stride/avgpool).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int = 1,
    ) -> None:
        super().__init__()

        layers = []
        if upsample_factor > 1:
            layers.append(
                nn.Upsample(
                    scale_factor=upsample_factor,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        layers.extend(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

        self.skip: nn.Module
        if in_channels != out_channels or upsample_factor > 1:
            skip_layers: list[nn.Module] = []
            if upsample_factor > 1:
                skip_layers.append(
                    nn.Upsample(
                        scale_factor=upsample_factor,
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            skip_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                ]
            )
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


class GMR_ResNetDecoder(nn.Module):
    """ResNet-style decoder that mirrors the GMR_ResNet encoder architecture.

    Reverses the spatial and channel transformations of the encoder using
    bilinear upsampling and standard convolutions with residual connections.

    Architecture (mirroring encoder with default settings)::

        z (input_dim)
        → fc → reshape  (8*inplanes × seed × seed)
        → layer4  (8*inplanes → 4*inplanes, upsample 2×)
        → layer3  (4*inplanes → 2*inplanes, upsample 2×)
        → layer2  (2*inplanes → inplanes,   upsample 2×)
        → layer1  (inplanes → inplanes)
        → upsample 2×  (reverse first avgpool)
        → conv_out → Sigmoid  (inplanes → out_channels)

    Args:
        input_dim: Latent vector size (must match encoder ``num_classes``).
        output_dim: Target image shape ``[C, H, W]``.
        inplanes: Base channel width (must match encoder ``inplanes``).
        layers: Number of residual blocks per stage, in encoder order
            ``[layer1, layer2, layer3, layer4]``.
        layer_stride: Per-stage stride used by the encoder. The decoder
            reverses each stride > 1 with bilinear upsampling.
        skip_first_maxpool: Must match the encoder setting. When *False*
            (default) the encoder applies an initial 2× avg-pool, which
            the decoder undoes with an extra 2× upsample.
        weights: Optional pre-trained weights.
        freeze: Freeze all parameters after initialisation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: list[int],
        inplanes: int = 64,
        layers: list[int] = [2, 2, 2, 2],
        layer_stride: list[int] = [1, 2, 2, 2],
        skip_first_maxpool: bool = False,
        weights: Optional[WeightsProvider] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        out_channels, out_h, out_w = output_dim

        self.example_input_array = torch.randn(1, input_dim)

        # Compute the spatial size at the deepest encoder feature map.
        # The encoder halves with an initial avg-pool (unless skip_first_maxpool)
        # then applies each layer_stride.
        seed_size_h = out_h
        seed_size_w = out_w
        if not skip_first_maxpool:
            seed_size_h //= 2
            seed_size_w //= 2
        for s in layer_stride:
            seed_size_h //= s
            seed_size_w //= s

        top_channels = 8 * inplanes  # encoder's deepest channels

        # -- fc: latent → spatial feature map --------------------------------
        self.fc = nn.Sequential(
            nn.Linear(input_dim, top_channels * seed_size_h * seed_size_w),
            nn.Unflatten(1, (top_channels, seed_size_h, seed_size_w)),
            nn.BatchNorm2d(top_channels),
            nn.ReLU(inplace=True),
        )

        # -- decoder residual layers (reverse order of encoder) ---------------
        # Encoder stages: layer1(inplanes), layer2(2*ip), layer3(4*ip), layer4(8*ip)
        # Decoder mirrors: layer4_dec, layer3_dec, layer2_dec, layer1_dec
        channel_schedule = [inplanes, 2 * inplanes, 4 * inplanes, 8 * inplanes]

        decoder_stages: list[nn.Module] = []
        for stage_idx in reversed(range(4)):
            in_ch = channel_schedule[stage_idx] if stage_idx == 3 else channel_schedule[stage_idx + 1]
            # After the first block of this stage, output channels is the
            # target for this stage (mirroring encoder's planes for this layer).
            out_ch = channel_schedule[stage_idx]
            stride = layer_stride[stage_idx]
            n_blocks = layers[stage_idx]

            blocks: list[nn.Module] = []
            # Repeat blocks (at same channels) — mirrors encoder's extra blocks
            for _ in range(n_blocks - 1):
                blocks.append(_DecoderBasicBlock(in_ch, in_ch))
            # Final block: channel change + upsampling (mirrors encoder's first block)
            blocks.append(_DecoderBasicBlock(in_ch, out_ch, upsample_factor=stride))
            decoder_stages.append(nn.Sequential(*blocks))

        self.decoder_layers = nn.Sequential(*decoder_stages)

        # -- reverse initial avg-pool -----------------------------------------
        if skip_first_maxpool:
            self.final_upsample: nn.Module = nn.Identity()
        else:
            self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # -- final projection (mirrors encoder conv1) -------------------------
        self.conv_out = nn.Sequential(
            nn.Conv2d(inplanes, out_channels, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid(),
        )

        if weights is not None:
            self.load_state_dict(weights.get_state_dict())
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.decoder_layers(x)
        x = self.final_upsample(x)
        x = self.conv_out(x)
        return x
