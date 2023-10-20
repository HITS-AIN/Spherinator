from data import IllustrisSdssDataModule
import torch

def test_empty():

    data = IllustrisSdssDataModule(["tests/data/"], num_workers=1)

    assert data.train_dataloader() == None

    try:
        data.setup("fit")
        assert False
    except ValueError:
        assert True


def test_fits():

    data = IllustrisSdssDataModule(["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
        num_workers=1)
    data.setup("fit")

    assert data.train_dataloader().batch_size == 32

    batch = next(iter(data.train_dataloader()))

    assert batch["image"].shape == (1, 3, 363, 363)
    assert batch["image"].dtype == torch.float32
    assert batch["image"].max() == 1.0
