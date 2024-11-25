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
    ):
        """Initializes the data set.

        Args:
            path (str): The data directory.
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the data. Defaults to None.
        """
        super().__init__()
        dataset = ds.dataset(path)
        scanner = dataset.scanner(batch_size=1)
        self.batches = scanner.to_batches()
        self.transform = transform

    def __iter__(self):
        for batch in self.batches:
            batch = torch.Tensor(batch.to_pydict()["data"])
            if self.transform is not None:
                batch = self.transform(batch)
            yield batch
