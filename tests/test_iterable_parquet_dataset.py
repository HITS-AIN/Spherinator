from torch.utils.data import DataLoader

from spherinator.data import IterableParquetDataset


def test_iterable_parquet_dataset(test_parquet_numpy_file):

    dataset = IterableParquetDataset(test_parquet_numpy_file)
    dataloader = DataLoader(dataset, num_workers=1)
    list(dataloader)
