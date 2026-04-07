from typing import List, Type, Union

import torch
import torch.nn as nn
from GMR_Conv import GMR_ResNet, GMRBasicBlock, GMRBottleneck


class GMRResNetEncoder(nn.Module):
    """Rotation-equivariant encoder using the GMR_ResNet backbone.

    Unlike ``GMR_ResNet`` used directly (which applies global average pooling
    and becomes rotation-**invariant**), this encoder stops before the
    ``avgpool`` step and retains a spatial feature map.  A local pooling to
    ``(latent_size, latent_size)`` and a 1×1 channel projection are applied
    instead, producing a ``(latent_channels, latent_size, latent_size)``
    feature map that transforms equivariantly under rotations, then flattened
    to a 1-D vector of size ``latent_channels * latent_size * latent_size``.

    Args:
        block: Block type (``GMRBasicBlock`` or ``GMRBottleneck``).
        layers: Number of blocks per stage, e.g. ``[2, 2, 2, 2]``.
        latent_channels: Output channels after the 1×1 projection. Default: 64.
        latent_size: Spatial size of the latent feature map. Default: 7.
        gmr_conv_size: GMR kernel size passed to ``GMR_ResNet``. Default: 9.
        All remaining keyword arguments are forwarded to ``GMR_ResNet``.
    """

    def __init__(
        self,
        block: Type[Union[GMRBasicBlock, GMRBottleneck]],
        layers: List[int],
        latent_channels: int = 64,
        latent_size: int = 7,
        gmr_conv_size: Union[int, list] = 9,
        **kwargs,
    ) -> None:
        super().__init__()

        # Build the full backbone (num_classes=1 is a placeholder; fc is never used)
        self._backbone = GMR_ResNet(
            block=block,
            layers=layers,
            gmr_conv_size=gmr_conv_size,
            num_classes=1,
            **kwargs,
        )

        # After layer4 the feature map has 512 channels (BasicBlock) or
        # 512*4=2048 (Bottleneck); use backbone.inplanes which is updated to
        # that value after _make_layer calls.
        backbone_channels = self._backbone.inplanes  # e.g. 512

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((latent_size, latent_size)),
            nn.Conv2d(backbone_channels, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.output_dim = latent_channels * latent_size * latent_size

        self.example_input_array = torch.randn(1, 3, 224, 224)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self._backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.first_avgpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = self.proj(x)
        return x
