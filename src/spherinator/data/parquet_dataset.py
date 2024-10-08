""" Iterable dataset reading parquet files.
"""

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


class ParquetDataset(IterableDataset):
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
                transformations to modify the images. Defaults to None.
        """

        self.parquet_file = pq.ParquetFile(path)
        self.transform = transform

    def __iter__(self) -> torch.Tensor:
        for batch in self.parquet_file.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
