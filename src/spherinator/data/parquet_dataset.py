""" Iterable dataset reading parquet files.
"""

import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
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
        table = dataset.to_table(columns=[data_column])
        self.data = table.to_pandas()[data_column]
        if b"shape" in table.schema.metadata:
            shape_string = table.schema.metadata[b"shape"].decode("utf-8")
            shape = shape_string.replace("(", "").replace(")", "").split(",")
            shape = tuple(map(int, shape))
            self.data = self.data.apply(lambda x: x.reshape(shape))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        batch = torch.Tensor(self.data[index])
        if self.transform is not None:
            batch = self.transform(batch)
        return batch
