import re
from dataclasses import dataclass
from typing import List, Optional

import lightning.pytorch as pl
import torch


@dataclass
class ParamConfig:
    pattern: str
    value: Optional[float] = None
    max_value: Optional[float] = None
    freeze: bool = False


class ParamManager(pl.Callback):
    """Callback to manage parameters of a LightningModule based on specified configurations.

    Args:
        configs (List[ParamConfig]): List of parameter configurations. Each configuration includes:
            - pattern: A regex pattern to match parameter names.
            - value: An optional float to set the parameter values to (if None, values are unchanged).
            - max_value: An optional float to clamp the parameter values to a maximum after each batch.
            - freeze: A boolean indicating whether to freeze the parameter (if True, requires_grad is set to False).
    """

    def __init__(self, configs: List[ParamConfig]):
        """
        The CLI will now strictly enforce that 'configs' is a list of
        objects with 'pattern', 'value', and 'freeze' fields.
        """
        self.configs = configs

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for conf in self.configs:
            pattern = re.compile(conf.pattern)

            for name, param in pl_module.named_parameters():
                if pattern.search(name):
                    # 1. Set Value
                    if conf.value is not None:
                        with torch.no_grad():
                            param.fill_(conf.value)
                        print(f"INFO: {name} set to {conf.value}")

                    # 2. Set Freeze State
                    param.requires_grad = not conf.freeze

                    # 3. Log max_value constraint
                    if conf.max_value is not None:
                        print(f"INFO: {name} max_value: {conf.max_value}")

                    state = "FROZEN" if conf.freeze else "TRAINABLE"
                    print(f"INFO: {name} status: {state}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        for conf in self.configs:
            if conf.max_value is None:
                continue
            pattern = re.compile(conf.pattern)
            for name, param in pl_module.named_parameters():
                if pattern.search(name):
                    with torch.no_grad():
                        param.clamp_(max=conf.max_value)
