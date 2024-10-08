from spherinator.data import ParquetDataset


def test_parquet_dataset():
    dataset = ParquetDataset(
        "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/parquet/0.parquet"
    )
    assert len(dataset) == 2
    data = dataset[0]
    assert data.shape == (3, 224, 224)
