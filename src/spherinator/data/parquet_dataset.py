"""Iterable dataset reading parquet files."""

from typing import Union

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
    """Iterable dataset reading parquet files."""

    def __init__(
        self,
        data_directory: str,
        data_column: Union[str, list[str]],
        transform=None,
        with_index: bool = False,
    ):
        """Initializes the data set.

        Args:
            data_directory (str): The data directory.
            data_column (str | list[str]): The column name(s) containing the data.
                The data columns will be merged using a list of strings.
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the data. Defaults to None.
            with_index (bool, optional): Whether to return the index with the data.
        """
        super().__init__()

        if not isinstance(data_column, list):
            data_column = [data_column]

        dataset = ds.dataset(data_directory)
        table = dataset.to_table(columns=data_column)
        self.transform = transform
        self.with_index = with_index

        if len(data_column) == 1:
            self.data = table[0].to_pandas()

            # Reshape the data if the shape is stored in the metadata.
            metadata_shape = bytes(data_column[0], "utf8") + b"_shape"
            if table.schema.metadata and metadata_shape in table.schema.metadata:
                shape_string = table.schema.metadata[metadata_shape].decode("utf8")
                shape = shape_string.replace("(", "").replace(")", "").split(",")
                shape = tuple(map(int, shape))
                self.data = self.data.apply(lambda x: x.reshape(shape))
        else:
            data = table.to_pandas()
            data["concat"] = data.apply(
                lambda x: np.concatenate((x[data_column[0]], x[data_column[1]])), axis=1
            )
            self.data = data["concat"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Union[torch.Tensor, tuple[torch.Tensor, int]]:
        batch = torch.tensor(self.data[index])
        if self.transform is not None:
            batch = self.transform(batch)
        if self.with_index:
            return batch, index
        else:
            return batch
