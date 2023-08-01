from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

import GalaxyZooDataModule
import models

def cli_main():
    cli = LightningCLI(models.RotationalSphericalProjectingAutoencoder,
                       GalaxyZooDataModule.GalaxyZooDataModule)


if __name__ == "__main__":
    cli_main()
