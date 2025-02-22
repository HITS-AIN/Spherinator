import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from spherinator.data import ParquetDataModule, ParquetDataset, ParquetDatasetSampling


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


def test_pyarray_to_numpy(parquet_numpy_file):
    """Test converting a pyarrow table to a numpy array."""
    dataset = ds.dataset(parquet_numpy_file)
    batch_size = 2
    scanner = dataset.scanner(batch_size=batch_size)
    batches = scanner.to_batches()
    batch = next(batches)
    batch = batch.column("data").to_numpy(zero_copy_only=False)

    assert len(batch) == batch_size
    assert (batch[0] == [5, 0, 3]).all()
    assert (batch[1] == [3, 7, 9]).all()


@pytest.mark.parametrize(
    ("batch_size", "num_workers"),
    [(1, 1), (1, 2), (2, 1), (2, 2)],
)
def test_parquet_dataset(parquet_numpy_file, batch_size, num_workers):
    """Test the ParquetDataset class."""
    dataset = ParquetDataset(parquet_numpy_file, data_column="data")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    assert len(iter(dataloader)) == len(dataset) / batch_size

    batch = next(iter(dataloader))
    assert batch.shape == (batch_size, 3)


def test_parquet_data_module_1d(parquet_1d_metadata):
    """Test the ParquetDataModule class with 1d data."""
    data = ParquetDataModule(
        parquet_1d_metadata,
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

    assert batch.shape == (2, 1, 12)
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


def test_parquet(parquet_test_merge):
    """Test the ParquetDataset with a single data columns."""
    dataset = ParquetDataset(parquet_test_merge, data_column="data1")
    dataloader = DataLoader(dataset)

    assert len(iter(dataloader)) == len(dataset)

    batch = next(iter(dataloader))
    assert (batch == torch.tensor([1.0, 2.0])).all()


def test_parquet_merge(parquet_test_merge):
    """Test the ParquetDataset with two merged data columns."""
    dataset = ParquetDataset(parquet_test_merge, data_column=["data1", "data2"])
    dataloader = DataLoader(dataset)

    assert len(iter(dataloader)) == len(dataset)

    batch = next(iter(dataloader))
    assert (batch == torch.tensor([1.0, 2.0, 3.0, 4.0])).all()


def test_absmax_norm(parquet_test_norm):
    """Test the ParquetDataModule normalization absmax."""
    data = ParquetDataModule(
        parquet_test_norm,
        data_column="data",
        normalize="absmax",
        batch_size=2,
        num_workers=1,
        shuffle=False,
    )
    data.setup("fit")
    dataloader = data.train_dataloader()
    batch = next(iter(dataloader))

    assert batch.shape == (2, 3)
    print(batch)
    assert torch.isclose(
        batch,
        torch.tensor([[-0.3077, 1.0000, -0.1282], [0.2073, -1.0000, -0.0488]]),
        atol=1e-3,
    ).all()


def test_parquet_dataset_with_index(parquet_numpy_file):
    """Test the ParquetDataset with returning the data index."""
    dataset = ParquetDataset(parquet_numpy_file, data_column="data", with_index=True)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    _, index = next(iter(dataloader))

    assert (index == torch.tensor([0, 1])).all()


def test_parquet_dataset_sampling(parquet_test_sampling):
    """Test the ParquetDataset with error sampling."""
    dataset = ParquetDatasetSampling(
        parquet_test_sampling, data_column="flux", error_column="flux_error"
    )
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    batch = next(iter(dataloader))

    assert batch.shape == (1, 4)
    print(batch)
    assert torch.isclose(
        batch,
        torch.tensor([[1.1509, 4.3388, 0.5688, 2.7741]]),
        atol=1e-3,
    ).all()
