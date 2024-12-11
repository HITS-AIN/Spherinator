""" Iterable dataset reading parquet files.
"""

import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset


class ParquetIterableDataset(IterableDataset):
    """Iterable dataset reading parquet files."""

    def __init__(
        self,
        data_directory: str,
        data_column: str = "data",
        batch_size: int = 64,
        transform=None,
    ):
        """Initializes the data set.

        Args:
            data_directory (str): The data directory.
            data_column (str, optional): The column name in the parquet file
                that contains the data. Defaults to "data".
            batch_size (int): The batch size. Defaults to 64.
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the data. Defaults to None.
        """
        super().__init__()
        self.data_column = data_column
        dataset = ds.dataset(data_directory)
        self.scanner = dataset.scanner(batch_size=batch_size)
        self.transform = transform

        self.shape = None
        metadata_shape = bytes(data_column, "utf8") + b"_shape"
        if metadata_shape in dataset.schema.metadata:
            shape_string = dataset.schema.metadata[metadata_shape].decode("utf8")
            shape = shape_string.replace("(", "").replace(")", "").split(",")
            self.shape = tuple(map(int, shape))

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        iterator = self.scanner.to_batches()

        # Special treatment for the multi-worker case
        # Idea adapted from itertools.islice
        # https://docs.python.org/3/library/itertools.html#itertools.islice
        next_i = 0
        step = 1
        if worker_info is not None:
            next_i = worker_info.id
            step = worker_info.num_workers

        for i, batch in enumerate(iterator):
            if i == next_i:
                next_i += step
                batch = batch.to_pandas()[self.data_column]
                batch = torch.Tensor(batch)
                if self.shape is not None:
                    batch = batch.reshape((batch.shape[0], *self.shape))
                if self.transform is not None:
                    batch = self.transform(batch)

                for item in batch:
                    yield item
