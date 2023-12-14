import pytest
import torch


@pytest.hookimpl
def pytest_runtest_setup():
    torch.manual_seed(0)
