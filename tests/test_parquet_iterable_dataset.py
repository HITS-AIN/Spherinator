import pytest
import torch
from torch.utils.data import DataLoader

from spherinator.data import ParquetIterableDataModule, ParquetIterableDataset


@pytest.mark.parametrize(("num_workers"), [1, 2])
def test_parquet_dataset(parquet_test_metadata, num_workers):
    """Test the ParquetIterableDataset class."""
    dataset = ParquetIterableDataset(parquet_test_metadata, batch_size=4)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=num_workers)

    iterator = iter(dataloader)
    batch = next(iterator)
    assert batch.shape == (2, 1)
    assert (batch == torch.Tensor([[0.0], [1.0]])).all()


@pytest.mark.parametrize(("num_workers"), [1, 2])
def test_parquet_data_module_1d(parquet_1d_metadata, num_workers):
    """Test the ParquetIterableDataModule class with 1d data."""
    data = ParquetIterableDataModule(
        parquet_1d_metadata,
        data_column="data",
        batch_size=2,
        scanner_batch_size=8,
        num_workers=num_workers,
    )
    data.setup("fit")
    dataloader = data.train_dataloader()

    assert dataloader.batch_size == 2
    assert dataloader.num_workers == num_workers

    batch = next(iter(dataloader))

    assert batch.shape == (2, 1, 12)
    assert batch.dtype == torch.float32

    assert len(list(dataloader)) == 5
