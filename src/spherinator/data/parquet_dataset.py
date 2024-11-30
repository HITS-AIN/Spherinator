""" Iterable dataset reading parquet files.
"""

import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset


class ParquetDataset(IterableDataset):
    """Iterable dataset reading parquet files."""

    def __init__(
        self,
        data_directory: str,
        data_column: str = "data",
        transform=None,
    ):
        """Initializes the data set.

        Args:
            path (str): The data directory.
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the data. Defaults to None.
        """
        super().__init__()
        self.data_column = data_column
        dataset = ds.dataset(data_directory)
        scanner = dataset.scanner(batch_size=1)
        self.batches = scanner.to_batches()
        self.transform = transform

    def __iter__(self):
        for batch in self.batches:
            batch = torch.Tensor(batch.to_pydict()[self.data_column])
            if self.transform is not None:
                batch = self.transform(batch)
            yield batch
