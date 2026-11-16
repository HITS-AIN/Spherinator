"""Per-sample tensor transforms, usable as ``Column`` transforms of the :class:`DataModule`."""

import torch


class MinMaxNormalize:
    """Normalize a tensor to [0, 1], treating constant tensors as zero."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        denominator = (tensor.max() - tensor.min()).clamp_min(torch.finfo(tensor.dtype).eps)
        return (tensor - tensor.min()) / denominator


class AbsMaxNormalize:
    """Normalize a tensor by its largest absolute value."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        denominator = tensor.abs().max().clamp_min(torch.finfo(tensor.dtype).eps)
        return tensor / denominator
