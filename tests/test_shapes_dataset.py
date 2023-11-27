from random import shuffle
from torch.utils.data import DataLoader

from data import ShapesDataset


def test_dataset():
    dataset = ShapesDataset("tests/data/shapes")

    assert len(dataset) == 4000

    data, index = dataset[0]

    assert index == 0
    assert data.shape == (64, 64)
    assert dataset.get_metadata(index)["index"] == 0


def test_dataloader():
    dataset = ShapesDataset("tests/data/shapes")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch, index = next(iter(dataloader))

    print(index)

    assert batch.shape == (2, 64, 64)
    assert dataset.get_metadata(index)["index"][0] == [0, 1]
