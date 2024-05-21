import numpy as np
import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from data import ShapesDataset


def test_dataset(shape_path):
    dataset = ShapesDataset(shape_path)

    assert len(dataset) == 4

    data = dataset[0]
    index = dataset.current_index

    assert index == 0
    assert data.shape == (64, 64)
    assert dataset.get_metadata(index)["filename"] == str(shape_path / "boxes.npy")


def test_dataloader(shape_path):
    dataset = ShapesDataset(shape_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch = next(iter(dataloader))

    assert batch.shape == (2, 64, 64)


def test_batch_sampler(shape_path):
    dataset = ShapesDataset(shape_path)
    sampler = BatchSampler(RandomSampler(dataset), batch_size=3, drop_last=False)
    loader = DataLoader(dataset, sampler=sampler)

    batch = next(iter(loader))
    assert batch.shape == (1, 3, 64, 64)
    assert dataset.current_index == [2, 1, 3]


def test_metadata(shape_path):
    dataset = ShapesDataset(shape_path)
    assert dataset.get_metadata(1)["filename"] == str(shape_path / "boxes.npy")
    assert dataset.get_metadata(2)["filename"] == str(shape_path / "circles.npy")


@pytest.mark.xfail(
    reason="Python builtin <built-in function empty> is currently not supported in Torchscript"
)
def test_jit():
    dataset = ShapesDataset("tests/data/shapes")
    torch.jit.script(dataset)
