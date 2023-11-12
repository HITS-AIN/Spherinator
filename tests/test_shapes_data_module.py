import torch

from data import ShapesDataModule


def test_fit():

    data = ShapesDataModule("tests/data/shapes", num_workers=1)
    data.setup("fit")

    assert data.train_dataloader().batch_size == 32

    batch = next(iter(data.train_dataloader()))

    assert batch["image"].shape == (32, 3, 91, 91)
    assert batch["image"].dtype == torch.float32
    assert batch["image"].max() == 0.9668
