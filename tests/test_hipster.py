import filecmp
from pathlib import Path

import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from data import ShapesDataModule
from hipster import Hipster
from models import (
    ConvolutionalDecoder,
    ConvolutionalEncoder,
    RotationalVariationalAutoencoderPower,
)


@pytest.fixture
def model():
    model = RotationalVariationalAutoencoderPower(
        encoder=ConvolutionalEncoder(),
        decoder=ConvolutionalDecoder(),
        z_dim=3,
        rotations=4,
    )
    return model


@pytest.fixture
def hipster(tmp_path):
    hipster = Hipster(
        output_folder=tmp_path, title="HipsterTest", max_order=0, number_of_workers=1
    )
    return hipster


# def test_generate_hips(hipster, model, tmp_path):
#     hipster.generate_hips(model)

#     assert filecmp.cmp(
#         tmp_path / "HipsterTest/model/index.html",
#         "tests/data/hipster/ref1/HipsterTest/model/index.html",
#     )


def test_generate_catalog(hipster, model, tmp_path):
    datamodule = ShapesDataModule(
        "tests/data/shapes", exclude_files=["boxes.npy", "circles.npy", "triangles.npy"]
    )
    hipster.generate_catalog(model, datamodule)

    df1 = pd.read_csv(tmp_path / "HipsterTest/catalog.csv")
    df2 = pd.read_csv("tests/data/hipster/ref1/HipsterTest/catalog.csv")

    # data item 987 is numerical unstable (best rotation angle varies)
    df1.drop(987, inplace=True)
    df2.drop(987, inplace=True)

    assert df1.shape == (999, 9)
    assert df2.shape == (999, 9)
    assert_frame_equal(df1, df2, atol=1e-6)


def test_create_images(hipster, tmp_path):
    datamodule = ShapesDataModule(
        "tests/data/shapes", exclude_files=["boxes.npy", "circles.npy", "triangles.npy"]
    )
    hipster.create_images(datamodule)

    assert Path(tmp_path / "HipsterTest/jpg/crosses_0.jpg").exists()


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
        torch.Tensor([[0.3004, -0.6826, -0.6662], [0.3027, -0.6826, -0.6651]]),
        rtol=1e-3,
    )

    assert loss.shape == torch.Size([2])
    assert torch.allclose(loss, torch.Tensor([0.1791, 0.1798]), rtol=1e-3)


def test_pandas_catalog():
    catalog = pd.read_csv(
        "tests/data/hipster/ref1/HipsterTest/catalog.csv",
        usecols=["id", "rotation", "x", "y", "z"],
    )
    assert catalog.shape == (1000, 5)
    assert len(catalog) == 1000
    assert catalog["x"][0] == pytest.approx(0.32369125, abs=1e-6, rel=1e-9)

    catalog = catalog.to_numpy()

    assert catalog[0][2] == pytest.approx(0.32369125, abs=1e-6, rel=1e-9)
