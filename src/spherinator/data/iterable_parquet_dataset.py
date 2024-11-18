""" Iterable dataset reading parquet files.
"""

import pyarrow.dataset as ds
import torch
from torch.multiprocessing import Queue
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
                transformations to modify the images. Defaults to None.
        """

        dataset = ds.dataset(path)
        self.transform = transform
        self.batches = Queue()
        [self.batches.put(batch) for batch in dataset.to_batches()]

    def __iter__(self):
        while True:
            if self.batches.empty() == True:
                self.batches.close()
                break

            batch = self.batches.get().to_pydict()
            if self.transform is not None:
                batch.update(self.transform(batch))
            yield batch
