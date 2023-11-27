import torch
import numpy as np

from data import IllustrisSdssDataModule


def test_empty():
    data = IllustrisSdssDataModule(["tests/data/"], num_workers=1)

    assert data.train_dataloader() == None

    try:
        data.setup("fit")
        assert False
    except ValueError:
        assert True


def test_fits():
    data = IllustrisSdssDataModule(
        ["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
        num_workers=1,
    )
    data.setup("fit")

    dataloader = data.train_dataloader()
    assert dataloader.batch_size == 32

    batch = next(iter(dataloader))

    assert batch["image"].shape == (1, 3, 363, 363)
    assert batch["image"].dtype == torch.float32
    assert batch["filename"] == [
        "tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/broadband_99185.fits"
    ]

    assert np.isclose(batch["image"].min(), 0.0)
    assert np.isclose(batch["image"].max(), 1.0)
