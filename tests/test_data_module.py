import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from spherinator.data import DataModule


def _parquet_file(tmp_path, size=20):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"data": [np.array([i], dtype=np.float32) for i in range(size)]}), path)
    return path


def test_no_validation_split_keeps_test_disjoint_from_training(tmp_path):
    data = DataModule(str(_parquet_file(tmp_path)), validation_size=0.0, test_size=0.25, batch_size=20)
    data.setup()

    train = {float(value) for batch in data.train_dataloader() for value in batch.flatten()}
    test = {float(value) for batch in data.test_dataloader() for value in batch.flatten()}

    assert len(train) == 15
    assert len(test) == 5
    assert train.isdisjoint(test)


def test_default_single_column_batch_is_tensor(tmp_path):
    data = DataModule(str(_parquet_file(tmp_path)), validation_size=0.2, test_size=0.0)
    data.setup("fit")

    assert isinstance(next(iter(data.train_dataloader())), torch.Tensor)


def _shaped_parquet_dir(tmp_path, files=2, rows=10):
    """Directory of parquet files with a flat 12-element ``data`` column and ``data_shape: (1,12)`` metadata."""
    directory = tmp_path / "ds"
    directory.mkdir()
    for i in range(files):
        table = pa.table(
            {"data": [np.random.rand(12).astype(np.float32) for _ in range(rows)]},
            metadata={"data_shape": "(1,12)"},
        )
        pq.write_table(table, directory / f"part-{i}.parquet")
    return directory


@pytest.mark.parametrize(
    ("path", "rows"),
    [
        ("ds/part-0.parquet", 10),
        ("ds", 20),
        ("ds/*.parquet", 20),
        ("ds/**/*.parquet", 20),
    ],
    ids=["file", "dir", "glob", "recursive-glob"],
)
def test_shape_metadata_is_applied_for_every_path_form(tmp_path, path, rows):
    _shaped_parquet_dir(tmp_path)
    data = DataModule(str(tmp_path / path), validation_size=0.0, test_size=0.0, batch_size=5)
    data.setup("fit")

    assert len(data._train_ds) == rows
    assert next(iter(data.train_dataloader())).shape == (5, 1, 12)


@pytest.mark.parametrize(
    ("validation_size", "test_size"),
    [(-0.1, 0.0), (1.0, 0.0), (0.2, -0.1), (0.2, 1.0)],
)
def test_invalid_split_sizes_are_rejected(validation_size, test_size):
    with pytest.raises(ValueError):
        DataModule("unused", validation_size=validation_size, test_size=test_size)
