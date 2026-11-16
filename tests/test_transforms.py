import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from spherinator.data import AbsMaxNormalize, DataModule, MinMaxNormalize


def _data_module(path, transform, batch_size=2):
    return DataModule(
        str(path),
        columns=[{"name": "data", "transform": transform}],
        validation_size=0.0,
        test_size=0.0,
        batch_size=batch_size,
        shuffle=False,
    )


def test_absmax_norm(parquet_test_norm):
    """Each sample is divided by its largest absolute value."""
    data = _data_module(parquet_test_norm, AbsMaxNormalize())
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))

    assert batch.shape == (2, 3)
    assert torch.isclose(
        batch,
        torch.tensor([[-0.3077, 1.0000, -0.1282], [0.2073, -1.0000, -0.0488]]),
        atol=1e-3,
    ).all()


def test_minmax_norm(parquet_test_norm):
    """Each sample is scaled to [0, 1]."""
    data = _data_module(parquet_test_norm, MinMaxNormalize())
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))

    assert batch.shape == (2, 3)
    assert torch.isclose(
        batch,
        torch.tensor([[0.0, 1.0, 0.1373], [1.0, 0.0, 0.7879]]),
        atol=1e-3,
    ).all()


@pytest.mark.parametrize("transform", [MinMaxNormalize(), AbsMaxNormalize()], ids=["minmax", "absmax"])
def test_normalization_of_constant_sample_is_finite(tmp_path, transform):
    path = tmp_path / "constant.parquet"
    pq.write_table(pa.table({"data": [np.ones(3, dtype=np.float32)]}), path)
    data = _data_module(path, transform, batch_size=1)
    data.setup("fit")

    batch = next(iter(data.train_dataloader()))

    assert torch.isfinite(batch).all()
