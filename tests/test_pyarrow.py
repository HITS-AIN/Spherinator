import pyarrow.dataset as ds
import pyarrow.parquet as pq


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


def test_parquet_table_metadata(parquet_2d_metadata):
    """Test reading metadata from a parquet table."""
    table = pq.read_table(parquet_2d_metadata)
    assert table.schema.metadata[b"data_shape"] == b"(1,12,12)"
