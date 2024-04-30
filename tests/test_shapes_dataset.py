import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from data import ShapesDataset


def test_dataset():
    dataset = ShapesDataset("tests/data/shapes")

    assert len(dataset) == 4000

    data = dataset[0]
    index = dataset.current_index

    assert index == 0
    assert data.shape == (64, 64)
    assert dataset.get_metadata(index)["filename"] == "tests/data/shapes/boxes.npy"


def test_dataloader():
    dataset = ShapesDataset("tests/data/shapes")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch = next(iter(dataloader))

    assert batch.shape == (2, 64, 64)


def test_batch_sampler():
    dataset = ShapesDataset("tests/data/shapes")
    sampler = BatchSampler(RandomSampler(dataset), batch_size=3, drop_last=False)
    loader = DataLoader(dataset, sampler=sampler)

    batch = next(iter(loader))
    assert batch.shape == (1, 3, 64, 64)
    assert dataset.current_index == [3226, 772, 1401]


def test_metadata():
    dataset = ShapesDataset("tests/data/shapes")
    assert dataset.get_metadata(999)["filename"] == "tests/data/shapes/boxes.npy"
    assert dataset.get_metadata(1000)["filename"] == "tests/data/shapes/circles.npy"


@pytest.mark.xfail(
    reason="Python builtin <built-in function empty> is currently not supported in Torchscript"
)
def test_jit():
    dataset = ShapesDataset("tests/data/shapes")
    torch.jit.script(dataset)
