import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from spherinator.data import ParquetDataModule, ParquetDataset


def test_pyarray_table(parquet_numpy_file):
    dataset = ds.dataset(parquet_numpy_file)
    table = dataset.to_table()
    assert len(table) == 10
    assert table.shape == (10, 2)

    table = dataset.to_table(columns=["data"])
    assert len(table) == 10
    assert table.shape == (10, 1)

    data = table.to_pandas()["data"]

    assert data[0] == 0.0


def test_pyarray_dataset_scanner(parquet_file):
    dataset = ds.dataset(parquet_file)
    scanner = dataset.scanner(batch_size=2)
    batches = scanner.to_batches()
    assert len(list(batches)) == 5


def test_pyarray_to_pydict(parquet_numpy_file):
    dataset = ds.dataset(parquet_numpy_file)
    scanner = dataset.scanner(batch_size=2)
    batches = scanner.to_batches()
    batch = next(batches)
    batch = batch.to_pydict()["data"]

    assert np.shape(batch) == (2, 1)
    assert batch == [[0], [2]]


def test_parquet_dataset(parquet_numpy_file):

    dataset = ParquetDataset(parquet_numpy_file)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    batch = next(iter(dataloader))
    assert batch.shape == (2, 1)


def test_parquet_data_module_1d(parquet_numpy_file):
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

    assert batch.shape == (5, 3, 2)
    assert batch.dtype == torch.float32

    assert len(list(data.train_dataloader())) == 2


def test_parquet_table_metadata(parquet_2d_metadata):
    table = pq.read_table(parquet_2d_metadata)
    assert table.schema.metadata[b"shape"] == b"(3,2)"
