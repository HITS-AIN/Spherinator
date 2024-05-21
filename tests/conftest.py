import random

import numpy as np
import pytest
import torch


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
