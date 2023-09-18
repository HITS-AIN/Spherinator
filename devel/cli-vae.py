import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule

import models

torch.set_float32_matmul_precision('high')

def cli_main():
    cli = LightningCLI(models.VAE(latent_dim=32, input_height=28, input_width=28, input_channels=1, lr=0.0001, batch_size=32),
                       MNISTDataModule,
                       save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
