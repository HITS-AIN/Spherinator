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


@pytest.mark.parametrize(
    ("validation_size", "test_size"),
    [(-0.1, 0.0), (1.0, 0.0), (0.2, -0.1), (0.2, 1.0)],
)
def test_invalid_split_sizes_are_rejected(validation_size, test_size):
    with pytest.raises(ValueError):
        DataModule("unused", validation_size=validation_size, test_size=test_size)
