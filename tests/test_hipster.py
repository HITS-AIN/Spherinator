from pathlib import Path

import pandas as pd
import pytest
import torch

from hipster import Hipster
from spherinator.data import ShapesDataModule
from spherinator.models import (
    ConvolutionalDecoder,
    ConvolutionalEncoder,
    RotationalVariationalAutoencoderPower,
)


@pytest.fixture
def model():
    model = RotationalVariationalAutoencoderPower(
        encoder=ConvolutionalEncoder(latent_dim=256),
        decoder=ConvolutionalDecoder(latent_dim=256),
        z_dim=3,
        rotations=4,
    )
    return model


@pytest.fixture
def hipster(tmp_path):
    hipster = Hipster(
        output_folder=tmp_path,
        title="HipsterTest",
        max_order=0,
        number_of_workers=1,
    )
    return hipster


def test_generate_hips(hipster, model, tmp_path):
    hipster.generate_hips(model)

    assert Path(tmp_path / "HipsterTest/model/index.html").exists()


def test_generate_catalog(hipster, model, tmp_path, shape_path):
    datamodule = ShapesDataModule(shape_path)
    hipster.generate_catalog(model, datamodule)

    df = pd.read_csv(
        tmp_path / "HipsterTest/catalog.csv", usecols=["id", "rotation", "x", "y", "z"]
    )

    assert df.shape == (4, 5)
    assert len(df) == 4
    assert df["x"][0] == pytest.approx(0.63545614, abs=1e-6, rel=1e-9)

    arr = df.to_numpy()

    assert arr[0][2] == pytest.approx(0.63545614, abs=1e-6, rel=1e-9)


def test_create_images(hipster, tmp_path, shape_path):
    datamodule = ShapesDataModule(shape_path, exclude_files=["boxes.npy"])
    hipster.create_images(datamodule)

    assert Path(tmp_path / "HipsterTest/jpg/circles_0.jpg").exists()


def test_contains_equal_element():
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 2]

    contains_equal_element = any(x in list1 for x in list2)

    assert contains_equal_element is True


def test_find_best_rotation(model):
    batch = torch.randn(2, 3, 4, 4)

    best_image, rot, coord, loss = model.find_best_rotation(batch)

    assert best_image.shape == torch.Size([2, 3, 128, 128])
    assert torch.allclose(rot, torch.Tensor([180.0, 270.0]), rtol=1e-3)

    assert coord.shape == torch.Size([2, 3])
    assert torch.allclose(
        coord,
        torch.Tensor([[0.6423, -0.7217, 0.2583], [0.6426, -0.7208, 0.2598]]),
        rtol=1e-3,
    )

    assert loss.shape == torch.Size([2])
    assert torch.allclose(loss, torch.Tensor([0.1814, 0.1820]), rtol=1e-3)
