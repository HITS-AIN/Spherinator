""" Iterable dataset reading parquet files.
"""

import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
    """Dataset reading parquet files."""

    def __init__(
        self,
        data_directory: str,
        data_column: str = "data",
        transform=None,
    ):
        """Initializes the data set.

        Args:
            data_directory (str): The data directory.
            data_column (str, optional): The column name in the parquet file
                that contains the data. Defaults to "data".
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the data. Defaults to None.
        """
        super().__init__()
        self.data_column = data_column
        self.dataset = ds.dataset(data_directory)
        self.len = self.dataset.count_rows()
        self.transform = transform

        self.shape = None
        metadata_shape = bytes(data_column, "utf8") + b"_shape"
        if metadata_shape in self.dataset.schema.metadata:
            shape_string = self.dataset.schema.metadata[metadata_shape].decode("utf8")
            shape = shape_string.replace("(", "").replace(")", "").split(",")
            self.shape = tuple(map(int, shape))

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> torch.Tensor:
        table = self.dataset.take([index], columns=[self.data_column])
        batch = table.to_pandas()[self.data_column]
        if self.shape is not None:
            batch = batch.apply(lambda x: x.reshape(self.shape))
        batch = torch.Tensor(batch[0])
        if self.transform is not None:
            batch = self.transform(batch)
        return batch
