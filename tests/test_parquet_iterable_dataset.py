import torch
from torch.utils.data import DataLoader

from spherinator.data import ParquetIterableDataset


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
