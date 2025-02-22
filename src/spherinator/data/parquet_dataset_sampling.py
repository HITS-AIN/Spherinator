"""Dataset reading parquet files."""

from typing import Union

import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset


class ParquetDatasetSampling(Dataset):
    """Dataset reading parquet files.

    The dataset will be sampled using a normal distribution using a 1-sigma deviation from the error column.
    Files with a leading underscore or dot will be ignored.
    """

    def __init__(
        self,
        data_directory: str,
        data_column: str,
        error_column: str,
        transform=None,
        with_index: bool = False,
    ):
        """Initialize ParquetDatasetSampling

        Args:
            data_directory (str): The data directory.
            data_column (str | list[str]): The column name(s) containing the data.
                The data columns will be merged using a list of strings.
            error_column (str, optional): The column name containing the error data.
                The error is in 1-sigma normal distribution. Defaults to None.
            transform (torchvision.transforms, optional): A single or a set of
                transformations to modify the data. Defaults to None.
            with_index (bool, optional): Whether to return the index with the data.
        """
        super().__init__()

        dataset = ds.dataset(
            data_directory, format="parquet", ignore_prefixes=["_", "."]
        )
        table = dataset.to_table(columns=[data_column, error_column])
        self.transform = transform
        self.with_index = with_index
        self.error_column = error_column

        self.data = table[0].to_pandas()

        # Reshape the data if the shape is stored in the metadata.
        metadata_shape = bytes(data_column[0], "utf8") + b"_shape"
        if table.schema.metadata and metadata_shape in table.schema.metadata:
            shape_string = table.schema.metadata[metadata_shape].decode("utf8")
            shape = shape_string.replace("(", "").replace(")", "").split(",")
            shape = tuple(map(int, shape))
            self.data = self.data.apply(lambda x: x.reshape(shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        batch = torch.normal(
            mean=torch.tensor(self.data["flux"][index], dtype=torch.float32),
            std=torch.tensor(self.data["flux_error"][index], dtype=torch.float32),
        )

        if self.transform is not None:
            batch = self.transform(batch)
        if self.with_index:
            return batch, torch.tensor(index)
        else:
            return batch
