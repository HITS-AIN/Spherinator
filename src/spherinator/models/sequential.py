from typing import List

import torch.nn as nn


class Sequential(nn.Sequential):
    """A drop-in for ``torch.nn.Sequential`` that accepts a ``modules`` keyword
    argument instead of ``*args``, making it compatible with jsonargparse /
    LightningCLI YAML configs.

    Example YAML usage::

        class_path: spherinator.models.Sequential
        init_args:
          modules:
            - class_path: spherinator.models.GMRResNetSpatialEncoder
              init_args: ...
            - class_path: torch.nn.AdaptiveAvgPool2d
              init_args:
                output_size: [1, 1]
    """

    def __init__(self, modules: List[nn.Module]) -> None:
        super().__init__(*modules)
