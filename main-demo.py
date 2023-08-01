from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)


if __name__ == "__main__":
    cli_main()
