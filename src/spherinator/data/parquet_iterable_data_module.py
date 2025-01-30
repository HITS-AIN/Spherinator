import torch
import torchvision.transforms.v2 as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from spherinator.data.parquet_iterable_dataset import ParquetIterableDataset


class ParquetIterableDataModule(LightningDataModule):
    """Defines access to the ParquetIterableDataset."""

    def __init__(
        self,
        data_directory: str,
        data_column: str = "data",
        batch_size: int = 32,
        scanner_batch_size: int = 512,
        num_workers: int = 1,
    ):
        """Initializes the data loader

        Args:
            data_directory (str): The data directory
            data_column (str, optional): The column to read from the parquet file. Defaults to "data".
            batch_size (int, optional): The batch size for training. Defaults to 32.
            scanner_batch_size (int, optional): The batch size for the parquet scanner. Defaults to 512.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
        """
        super().__init__()

        self.data_directory = data_directory
        self.data_column = data_column
        self.batch_size = batch_size
        self.scanner_batch_size = scanner_batch_size
        self.num_workers = num_workers
        self.data_train = None
        self.dataloader_train = None

        self.transform_train = transforms.Compose(
            [
                transforms.Lambda(  # Normalize
                    lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                )
            ]
        )

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
                         For the moment just fitting is supported.
        """

        if stage == "fit" and self.data_train is None:
            self.data_train = ParquetIterableDataset(
                data_directory=self.data_directory,
                data_column=self.data_column,
                batch_size=self.scanner_batch_size,
                transform=self.transform_train,
            )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        else:
            raise ValueError(f"Stage {stage} not supported.")

    def train_dataloader(self):
        """Gets the data loader for training."""
        return self.dataloader_train
