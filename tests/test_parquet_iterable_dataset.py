import torch
from torch.utils.data import DataLoader

from spherinator.data import ParquetIterableDataModule, ParquetIterableDataset


def test_parquet_dataset(parquet_test_metadata):
    """Test the ParquetIterableDataset class."""
    dataset = ParquetIterableDataset(parquet_test_metadata, batch_size=4)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    iterator = iter(dataloader)
    batch = next(iterator)
    assert batch.shape == (2, 1)
    assert (batch == torch.Tensor([[0.0], [1.0]])).all()

    batch = next(iterator)
    assert batch.shape == (2, 1)
    assert (batch == torch.Tensor([[2.0], [3.0]])).all()


def test_parquet_data_module_1d(parquet_1d_metadata):
    """Test the ParquetDataModule class with 1d data."""
    data = ParquetIterableDataModule(
        parquet_1d_metadata,
        data_column="data",
        batch_size=2,
        num_workers=1,
    )
    data.setup("fit")

    dataloader = data.train_dataloader()

    assert dataloader.batch_size == 2
    assert dataloader.num_workers == 1

    batch = next(iter(dataloader))

    assert batch.shape == (2, 1, 12)
    assert batch.dtype == torch.float32

    assert len(list(data.train_dataloader())) == 5
