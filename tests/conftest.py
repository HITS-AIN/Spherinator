import random
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from .utils import random_date, random_description, random_image


@pytest.hookimpl
def pytest_runtest_setup():
    """Seed all random number generators to be fully deterministic."""
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def shape_path(tmp_path_factory):
    """Mock data for the ShapesDataset."""
    path = tmp_path_factory.mktemp("data")
    np.save(path / "boxes.npy", np.random.random((2, 64, 64)))
    np.save(path / "circles.npy", np.random.random((2, 64, 64)))
    return path


@pytest.fixture(scope="session")
def parquet_file(tmp_path_factory):
    """Mock parquet data file."""
    series = []
    for _ in range(10):
        item = {
            "id": uuid4().hex,
            "timestamp": random_date("1/1/2008 1:30 PM", "1/1/2024 4:50 AM"),
            "description": random_description(),
            "image": random_image(),
        }
        series.append(pd.Series(item))

        df = pd.DataFrame(series)

    file = tmp_path_factory.mktemp("data") / "test.parquet"
    df.to_parquet(file)

    return file


@pytest.fixture(scope="session")
def parquet_numpy_file(tmp_path_factory):
    """Mock parquet data file containing 1-dim numpy array."""
    series = []
    for i in range(10):
        item = {
            "id": i,
            "data": np.array([2 * i], np.int32),
        }
        series.append(pd.Series(item))

        df = pd.DataFrame(series)

    file = tmp_path_factory.mktemp("data") / "test.parquet"
    df.to_parquet(file)

    return file


@pytest.fixture(scope="session")
def parquet_file_numpy_2d(tmp_path_factory):
    """Mock parquet data file containing 2-dim numpy array."""
    series = []
    for i in range(10):
        item = {
            "id": i,
            "data": np.random.rand(3, 2).flatten(),
        }
        series.append(pd.Series(item))

        df = pd.DataFrame(series)

    file = tmp_path_factory.mktemp("data") / "test.parquet"
    df.to_parquet(file)

    return file


@pytest.fixture(scope="session")
def parquet_2d_metadata(tmp_path_factory):
    """Mock parquet data with flatten 2d array and metadata."""

    table = pa.table(
        {"id": range(10), "data": [np.random.rand(3, 2).flatten() for _ in range(10)]},
        metadata={"shape": "(3,2)"},
    )

    file = tmp_path_factory.mktemp("data") / "test.parquet"
    pq.write_table(table, file)

    return file
