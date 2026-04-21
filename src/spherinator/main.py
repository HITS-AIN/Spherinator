#!/usr/bin/env python3
"""Uses the command line client to start the training."""

import os

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

# Raise a FloatingPointError for any kind of floating-point errors
if __debug__:
    np.seterr(all="raise")
    print("debug on")
else:
    print("debug off")


# Set the default precision of torch operations to float32
# See https://github.com/Lightning-AI/lightning/discussions/16698
torch.set_float32_matmul_precision("high")


class WandbSaveConfigCallback(SaveConfigCallback):
    """Custom SaveConfigCallback that saves the config to the wandb run's files directory."""

    def setup(self, trainer, pl_module, stage):
        pass  # Prevent saving to default log_dir before wandb is ready

    def on_train_start(self, trainer, pl_module):
        if self.already_saved:
            return
        log_dir = trainer.logger.experiment.dir
        config_path = os.path.join(log_dir, self.config_filename)
        self.parser.save(self.config, config_path, skip_none=False, overwrite=True)
        trainer.logger.experiment.save(config_path)
        self.already_saved = True


def main():
    LightningCLI(
        save_config_callback=WandbSaveConfigCallback, save_config_kwargs={"config_filename": "cli-config.yaml"}
    )


if __name__ == "__main__":
    main()
