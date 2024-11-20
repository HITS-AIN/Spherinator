""" Iterable dataset reading parquet files.
"""

import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset


class IterableParquetDataset(IterableDataset):
    """Iterable dataset reading parquet files."""

    def __init__(
        self,
        path: str,
        transform=None,
        batch_size: int = 1,
    ):
        """Initializes the data set.

        Args:
            path (str): The data directory.
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the data. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 1.
        """

        dataset = ds.dataset(path)
        scanner = dataset.scanner(batch_size=batch_size)
        self.batches = scanner.to_batches()
        self.transform = transform

    def __iter__(self):
        for batch in self.batches:
            batch = torch.Tensor(batch.to_pydict()["data"])
            if self.transform is not None:
                batch = self.transform(batch)
            yield batch
