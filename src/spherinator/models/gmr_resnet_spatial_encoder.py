from typing import Any, List, Optional, Type, Union

import torch
import torch.nn as nn
from GMR_Conv import GMR_ResNet, GMRBasicBlock, GMRBottleneck

from .weights_provider import WeightsProvider


class GMRResNetSpatialEncoder(GMR_ResNet):
    """GMR_ResNet variant that outputs a spatial feature map instead of a class vector.

    Removes the global-average-pool + flatten + Linear head and replaces it with
    a 1×1 convolution from the deepest feature map channels (``8 * inplanes``) to
    ``latent_channels``.  The encoder output is a 4-D tensor
    ``(B, latent_channels, H, W)`` where H × W is the spatial resolution of the
    final feature map (typically 14 × 14 for 224 × 224 input with default strides).

    This lets positional and phase structure encoded by the equivariant GMR layers
    flow into the latent space without being collapsed by global average pooling.

    Args:
        block: Block type (``GMRBasicBlock`` or ``GMRBottleneck``).
        layers: Number of residual blocks per stage ``[layer1, layer2, layer3, layer4]``.
        latent_channels: Number of output channels in the spatial latent tensor.
            Must match the paired ``GMRResNetDecoder`` ``input_channels``.
        inplanes: Base channel width.  Must match the paired decoder ``inplanes``.
        layer_stride: Per-stage stride (list of 4).
        skip_first_maxpool: When ``False`` (default) an initial 2× avg-pool is
            applied after ``conv1``.
        in_channels: Number of input image channels.
        weights: Optional pre-trained weights.
        freeze: Freeze all parameters after initialisation.
        **kwargs: Forwarded verbatim to :class:`GMR_ResNet` (e.g. ``gmr_conv_size``,
            ``num_rings``, ``sigma_no_weight_decay``).
    """

    def __init__(
        self,
        block: Type[Union[GMRBasicBlock, GMRBottleneck]],
        layers: List[int],
        input_dim: List[int],
        latent_channels: int = 64,
        inplanes: int = 64,
        layer_stride: Union[int, List[int]] = [1, 2, 2, 2],
        skip_first_maxpool: bool = False,
        weights: Optional[WeightsProvider] = None,
        freeze: bool = False,
        **kwargs: Any,
    ) -> None:
        # Build the base ResNet; num_classes=1 is a dummy placeholder —
        # we discard avgpool/flatten/fc entirely and replace with latent_conv.
        super().__init__(
            block=block,
            layers=layers,
            num_classes=1,
            inplanes=inplanes,
            layer_stride=layer_stride,
            skip_first_maxpool=skip_first_maxpool,
            in_channels=input_dim[0],
            **kwargs,
        )

        self.latent_channels = latent_channels
        top_channels = 8 * inplanes * block.expansion

        # Replace global-average-pool + flatten + fc with a channel-reducing 1×1 conv.
        # This keeps the full (H, W) spatial structure in the latent representation.
        self.latent_conv = nn.Sequential(
            nn.Conv2d(top_channels, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # example_input_array for Lightning model summary / ONNX export
        self.example_input_array = torch.randn(1, *input_dim)

        # Infer spatial latent shape from a dummy forward pass
        with torch.no_grad():
            dummy_out = self.forward(self.example_input_array)
        self.output_dim = tuple(dummy_out.shape[1:])  # (C, H, W)

        if weights is not None:
            self.load_state_dict(weights.get_state_dict())
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return spatial latent tensor ``(B, latent_channels, H, W)``."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.first_avgpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Skip avgpool / flatten / fc — use 1×1 conv instead
        x = self.latent_conv(x)
        return x  # (B, latent_channels, H, W)
