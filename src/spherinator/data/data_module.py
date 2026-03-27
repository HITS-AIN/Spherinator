from typing import Any, Callable, Optional

import lightning as L
import pyarrow.dataset as pa_ds
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class Column:
    """
    A class representing a data column with its name and associated transformations.

    Args:
        name: The name of the column
        transform: Optional transformation function to apply to the column data
        **kwargs: Additional parameters that can be stored with the column
    """

    def __init__(
        self,
        name: str,
        transform: Optional[Callable[[Any], Any]] = None,
        shape: Optional[tuple[int, ...]] = None,
        **kwargs,
    ):
        self.name = name
        self.transform = transform
        self.shape = shape

        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def apply_transform(self, data: Any) -> Any:
        """
        Apply the transformation to the given data.

        Args:
            data: The data to transform

        Returns:
            Transformed data if transform is available, otherwise original data
        """
        if isinstance(data, list):
            data = torch.tensor(data)
        shape = getattr(self, "shape", None)
        if shape is not None:
            data = data.reshape(shape)
        if self.transform is not None:
            if isinstance(data, list):
                return [self.transform(item) for item in data]
            else:
                return self.transform(data)
        return data

    def has_transform(self) -> bool:
        """Check if this column has a transformation function."""
        return self.transform is not None

    def get_metadata(self) -> dict[str, Any]:
        """Get all metadata associated with this column (excluding name and transform)."""
        metadata = {}
        for key, value in self.__dict__.items():
            if key not in ["name", "transform"]:
                metadata[key] = value
        return metadata

    def copy(self, **override_kwargs) -> "Column":
        """
        Create a copy of this column with optional parameter overrides.

        Args:
            **override_kwargs: Parameters to override in the copy

        Returns:
            A new Column instance
        """
        # Get all current attributes except name and transform
        attrs = {}
        for key, value in self.__dict__.items():
            if key not in ["name", "transform"]:
                attrs[key] = value

        # Get name and transform with potential overrides
        name = override_kwargs.pop("name", self.name)
        transform = override_kwargs.pop("transform", self.transform)

        # Add any remaining overrides
        attrs.update(override_kwargs)

        return Column(name, transform, **attrs)

    def __repr__(self) -> str:
        metadata_str = ""
        metadata = self.get_metadata()
        if metadata:
            metadata_items = [f"{k}={repr(v)}" for k, v in list(metadata.items())[:2]]  # Show first 2 metadata items
            if len(metadata) > 2:
                metadata_items.append(f"... +{len(metadata) - 2} more")
            metadata_str = f", {', '.join(metadata_items)}"

        return f"Column(name='{self.name}', transform={self.transform is not None}{metadata_str})"


class TransformedDataset(Dataset):
    """A PyTorch Dataset wrapper that applies column transformations."""

    def __init__(self, dataset, columns: list[Column], return_dict: bool):
        self.dataset = dataset
        self.columns = columns
        self.return_dict = return_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.return_dict:
            transformed_sample = {}
        else:
            transformed_sample = []

        for column in self.columns:
            if column.name in sample:
                transformed_column = column.apply_transform(sample[column.name])
                if self.return_dict:
                    transformed_sample[column.name] = transformed_column
                else:
                    transformed_sample.append(transformed_column)

        # If not returning dict and only one column, return the single value
        if not self.return_dict and len(transformed_sample) == 1:
            return transformed_sample[0]
        return transformed_sample


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        columns: Optional[list[Column]] = None,
        return_dict: bool = True,
        **dataloader_kwargs,
    ):
        super().__init__()

        if columns is None:
            columns = [Column(name="data")]

        self.path: str = path
        self.columns: list[Column] = columns
        self.return_dict: bool = return_dict
        self._dataset: Optional[TransformedDataset] = None

        # Store DataLoader kwargs for forwarding
        self.dataloader_kwargs = dataloader_kwargs

    def prepare_data(self):
        load_dataset(self.path)

    def setup(self, stage: str):
        if self._dataset is not None:
            return
        dataset = load_dataset(self.path)
        train_dataset = dataset["train"]
        train_dataset.set_format("torch")  # Ensure the dataset returns PyTorch tensors

        # Read parquet schema metadata and attach shape info to columns.
        pa_dataset = pa_ds.dataset(self.path, format="parquet", exclude_invalid_files=True)
        schema_meta = pa_dataset.schema.metadata or {}
        for column in self.columns:
            key = column.name.encode() + b"_shape"
            if key in schema_meta:
                parts = schema_meta[key].decode().strip("()").split(",")
                setattr(column, "shape", tuple(int(p) for p in parts if p.strip()))

        self._dataset = TransformedDataset(
            train_dataset,
            self.columns,
            return_dict=self.return_dict,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset,
            **self.dataloader_kwargs,
        )
