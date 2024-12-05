import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from spherinator.data import ParquetDataModule, ParquetDataset


def test_pyarray_dataset_scanner(parquet_file):
    """Test reading a parquet file."""
    dataset = ds.dataset(parquet_file)
    scanner = dataset.scanner(batch_size=2)
    batches = scanner.to_batches()
    assert len(list(batches)) == 5


def test_pyarray_table(parquet_numpy_file):
    """Test reading a parquet file with a numpy array."""
    dataset = ds.dataset(parquet_numpy_file)
    table = dataset.to_table()
    assert len(table) == 10
    assert table.shape == (10, 2)

    table = dataset.to_table(columns=["data"])
    assert len(table) == 10
    assert table.shape == (10, 1)

    data = table.to_pandas()["data"]
    assert data[0].shape == (3,)


def test_pyarray_to_pydict(parquet_numpy_file):
    """Test converting a pyarrow table to a dictionary."""
    dataset = ds.dataset(parquet_numpy_file)
    batch_size = 2
    scanner = dataset.scanner(batch_size=batch_size)
    batches = scanner.to_batches()
    batch = next(batches)
    batch = batch.to_pydict()["data"]
    assert len(batch) == batch_size
    assert batch[0] == [5, 0, 3]
    assert batch[1] == [3, 7, 9]


def test_parquet_dataset(parquet_numpy_file):
    """Test the ParquetDataset class."""
    dataset = ParquetDataset(parquet_numpy_file)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    batch = next(iter(dataloader))
    assert batch.shape == (2, 3)


def test_parquet_data_module_1d(parquet_numpy_file):
    """Test the ParquetDataModule class with 1d data."""
    data = ParquetDataModule(
        parquet_numpy_file,
        data_column="data",
        batch_size=2,
        num_workers=1,
        shuffle=True,
    )
    data.setup("fit")

    dataloader = data.train_dataloader()

    assert dataloader.batch_size == 2
    assert dataloader.num_workers == 1

    batch = next(iter(dataloader))

    assert batch.shape == (2, 3)
    assert batch.dtype == torch.float32

    assert len(list(data.train_dataloader())) == 5


def test_parquet_data_module_2d(parquet_2d_metadata):
    """Test the ParquetDataModule class with 2d data."""
    data = ParquetDataModule(
        parquet_2d_metadata,
        data_column="data",
        batch_size=5,
        num_workers=1,
        shuffle=True,
    )
    data.setup("fit")

    dataloader = data.train_dataloader()

    assert dataloader.batch_size == 5
    assert dataloader.num_workers == 1

    batch = next(iter(dataloader))

    assert batch.shape == (5, 1, 3, 2)
    assert batch.dtype == torch.float32

    assert len(list(data.train_dataloader())) == 2


def test_parquet_table_metadata(parquet_2d_metadata):
    """Test reading metadata from a parquet table."""
    table = pq.read_table(parquet_2d_metadata)
    assert table.schema.metadata[b"data_shape"] == b"(1,3,2)"
