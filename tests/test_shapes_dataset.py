import torch
from torch.utils.data import DataLoader

from data import ShapesDataset


def test_dataset():
    torch.manual_seed(0)
    dataset = ShapesDataset("tests/data/shapes")

    assert len(dataset) == 4000

    data, index = dataset[0]

    assert index == 0
    assert data.shape == (64, 64)
    assert dataset.get_metadata(index)["filename"] == "tests/data/shapes/boxes.npy"


def test_dataloader():
    torch.manual_seed(0)
    dataset = ShapesDataset("tests/data/shapes")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch, index = next(iter(dataloader))

    assert batch.shape == (2, 64, 64)
    assert index.tolist() == [3226, 772]
    assert (
        dataloader.dataset.get_metadata(index[0])["filename"]
        == "tests/data/shapes/triangles.npy"
    )
    assert (
        dataloader.dataset.get_metadata(index[1])["filename"]
        == "tests/data/shapes/boxes.npy"
    )
