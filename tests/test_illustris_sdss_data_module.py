import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import spherinator.data.preprocessing as pp
from spherinator.data import (
    IllustrisSdssDataModule,
    IllustrisSdssDataset,
    IllustrisSdssDatasetWithMetadata,
)


def test_empty():
    data = IllustrisSdssDataModule(["tests/data/"], num_workers=1)

    assert data.train_dataloader() == None

    try:
        data.setup("fit")
        assert False
    except ValueError:
        assert True


def test_dataloader():
    dataset = IllustrisSdssDataset(
        ["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
        transform=transforms.Compose(
            [
                transforms.Resize((200, 200), antialias=True),
                pp.CreateNormalizedRGBColors(
                    stretch=0.9,
                    range=5,
                    lower_limit=0.001,
                    channel_combinations=[[2, 3], [1, 0], [0]],
                    scalers=[0.7, 0.5, 1.3],
                ),
            ]
        ),
        minsize=10,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    assert len(dataloader) == 1

    batch = next(iter(dataloader))

    assert batch.shape == (2, 3, 200, 200)


def test_datamodule():
    data = IllustrisSdssDataModule(
        ["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
        num_workers=1,
        batch_size=4,
        shuffle=True,
        minsize=10,
    )
    data.setup("fit")

    dataloader = data.train_dataloader()
    assert dataloader.batch_size == 4

    batch = next(iter(dataloader))

    assert batch.shape == (2, 3, 363, 363)
    assert batch.dtype == torch.float32
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_metadata():
    dataset = IllustrisSdssDataset(
        ["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
    )
    assert (
        dataset.get_metadata(0)["filename"]
        == "tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/broadband_99185.fits"
    )
    assert dataset.get_metadata(0)["simulation"] == "TNG50"
    assert dataset.get_metadata(0)["snapshot"] == "095"
    assert dataset.get_metadata(0)["subhalo_id"] == "99185"


def test_dataloader_with_metadata():
    dataset = IllustrisSdssDatasetWithMetadata(
        ["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
        transform=IllustrisSdssDataModule([]).transform_processing,
        minsize=10,
    )
    dataloader = DataLoader(dataset, batch_size=2)

    batch, metadata = next(iter(dataloader))

    assert batch.shape == (2, 3, 363, 363)
    assert (
        metadata["filename"][0]
        == "tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/broadband_117378.fits"
    )


@pytest.mark.xfail(reason="'Tensor' object has no attribute or method '__init__'.")
def test_jit():
    datamodule = IllustrisSdssDataModule(
        ["tests/data/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/"],
        num_workers=1,
        batch_size=4,
    )
    torch.jit.script(datamodule)
