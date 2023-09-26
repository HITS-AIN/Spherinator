import torch
from lightning.pytorch.cli import LightningCLI

import data
import models

torch.set_float32_matmul_precision('high')

def cli_main():
    cli = LightningCLI(models.RotationalVariationalAutoencoder,
                       data.GalaxyZooDataModule,
                       save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
