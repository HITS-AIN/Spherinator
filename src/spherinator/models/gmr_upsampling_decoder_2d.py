from typing import Optional

from GMR_Conv.utils import convert_to_gmr_conv

from .upsampling_decoder_2d import UpsamplingDecoder2D
from .weights_provider import WeightsProvider


class GMRUpsamplingDecoder2D(UpsamplingDecoder2D):
    """UpsamplingDecoder2D with Conv2d layers replaced by GMR_Conv2d.

    Uses ``convert_to_gmr_conv`` to swap every Conv2d (kernel > 1×1)
    for a rotation/reflection-equivariant GMR_Conv2d kernel.

    Args:
        gmr_conv_size: Kernel size for the GMR convolutions.
            Passed through to ``convert_to_gmr_conv``.
        All other arguments are forwarded to ``UpsamplingDecoder2D``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: list[int],
        base_channels: int = 512,
        seed_size: int = 7,
        gmr_conv_size: int = 9,
        weights: Optional[WeightsProvider] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            base_channels=base_channels,
            seed_size=seed_size,
            weights=weights,
            freeze=freeze,
        )
        convert_to_gmr_conv(self, gmr_conv_size=gmr_conv_size)
