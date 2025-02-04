#!/usr/bin/env python3
"""Uses the command line client to start the training."""

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI

# Raise a FloatingPointError for any kind of floating-point errors
if __debug__:
    np.seterr(all="raise")
    print("debug on")
else:
    print("debug off")

# Set the default precision of torch operations to float32
# See https://github.com/Lightning-AI/lightning/discussions/16698
torch.set_float32_matmul_precision("high")


def main():
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
