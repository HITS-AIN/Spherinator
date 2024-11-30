import torch
import torchvision.transforms.v2 as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from spherinator.data.parquet_dataset import ParquetDataset


class ParquetDataModule(LightningDataModule):
    """Defines access to the ImagesDataset."""

    def __init__(
        self,
        data_directory: str,
        data_column: str = "data",
        shuffle: bool = True,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        """Initializes the data loader

        Args:
            data_directory (str): The data directory
            data_column (str, optional): The column to read from the parquet file. Defaults to "data".
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
        """
        super().__init__()

        self.data_directory = data_directory
        self.data_column = data_column
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train = None
        self.dataloader_train = None
        self.transform_train = None

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
                         For the moment just fitting is supported.
        """

        if stage == "fit" and self.data_train is None:
            self.data_train = ParquetDataset(
                data_directory=self.data_directory,
                data_column=self.data_column,
                transform=self.transform_train,
            )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
        else:
            raise ValueError(f"Stage {stage} not supported.")

    def train_dataloader(self):
        """Gets the data loader for training."""
        return self.dataloader_train
