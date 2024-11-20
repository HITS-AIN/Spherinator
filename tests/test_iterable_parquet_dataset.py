import pyarrow.dataset as ds
from torch.utils.data import DataLoader

from spherinator.data import IterableParquetDataset


def test_pyarray_dataset_scanner(parquet_file):
    dataset = ds.dataset(parquet_file)
    scanner = dataset.scanner(batch_size=2)
    batches = scanner.to_batches()
    assert len(list(batches)) == 5


def test_iterable_parquet_dataset(parquet_numpy_file):

    dataset = IterableParquetDataset(parquet_numpy_file, batch_size=2)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

    for batch in dataloader:
        print(batch)
        assert batch.shape == (1, 2, 1)
        break
