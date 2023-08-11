import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule

import models

torch.set_float32_matmul_precision('high')

def cli_main():
    cli = LightningCLI(models.SVAE(h_dim=256, z_dim=32, distribution='vmf'),
                       MNISTDataModule,
                       save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
