from typing import Optional, Union

import torch
import torchvision.transforms.v2 as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from spherinator.data.parquet_dataset import ParquetDataset
from spherinator.data.parquet_dataset_with_error import ParquetDatasetWithError


class ParquetDataModule(LightningDataModule):
    """Defines access to the ParquetDataset."""

    def __init__(
        self,
        data_directory: str,
        data_column: Union[str, list[str]],
        error_column: Optional[str] = None,
        normalize: Optional[str] = None,
        shuffle: bool = True,
        batch_size: int = 32,
        num_workers: int = 1,
        with_index: bool = False,
    ):
        """Initializes the data loader

        Args:
            data_directory (str): The data directory
            data_column (str | list[str]): The column name(s) containing the data.
                The data columns will be merged using a list of strings.
            error_column (str, optional): The column name containing the error data.
                The error is in 1-sigma normal distribution. Defaults to None.
            normalize (str, optional): The normalization (minmax, absmax, none) to apply. Defaults to "none".
            shuffle (bool, optional): Wether or not to shuffle whe reading. Defaults to True.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_workers (int, optional): How many worker to use for loading. Defaults to 1.
            with_index (bool, optional): Whether to return the index with the data.
        """
        super().__init__()

        self.data_directory = data_directory
        self.data_column = [data_column] if isinstance(data_column, str) else data_column
        self.error_column = error_column
        self.normalize = normalize
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.with_index = with_index

        self.data_train = None
        self.dataloader_train = None

        self.transform_train = None
        if normalize == "minmax":
            self.transform_train = transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)))
        elif normalize == "absmax":
            self.transform_train = transforms.Lambda(lambda x: x / torch.max(torch.abs(x)))

    def setup(self, stage: str):
        """Sets up the data set and data loaders.

        Args:
            stage (str): Defines for which stage the data is needed.
                         For the moment just fitting is supported.
        """
        if stage != "fit":
            raise ValueError(f"Stage {stage} not supported.")

        if stage == "fit" and self.data_train is None:
            if self.error_column is not None:
                self.data_train = ParquetDatasetWithError(
                    data_directory=self.data_directory,
                    data_column=self.data_column[0],
                    error_column=self.error_column,
                    transform=self.transform_train,
                    with_index=self.with_index,
                )
            else:
                self.data_train = ParquetDataset(
                    data_directory=self.data_directory,
                    data_column=self.data_column,
                    transform=self.transform_train,
                    with_index=self.with_index,
                )
            self.dataloader_train = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        """Gets the data loader for training."""
        return self.dataloader_train
