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
    torch.manual_seed(0)
    data = IllustrisSdssDataModule(
        ["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
        num_workers=4,
        batch_size=4,
        shuffle=True,
    )
    data.setup("fit")

    dataloader = data.train_dataloader()
    assert dataloader.batch_size == 4

    batch = next(iter(dataloader))

    assert batch["image"].shape == (4, 3, 363, 363)
    assert batch["image"].dtype == torch.float32

    print(batch["filename"])

    assert batch["filename"] == [
        "tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/broadband_117392.fits",
        "tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/broadband_117369.fits",
        "tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/broadband_117390.fits",
        "tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/broadband_117384.fits",
    ]

    assert batch["image"].min() >= 0.0
    assert batch["image"].max() <= 1.0
