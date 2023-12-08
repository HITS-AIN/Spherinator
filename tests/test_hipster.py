import filecmp

import pytest
import torch

from hipster import Hipster
from models import RotationalVariationalAutoencoderPower
from data import ShapesDataModule


@pytest.fixture
def model():
    model = RotationalVariationalAutoencoderPower(z_dim=3, rotations=4)
    return model


@pytest.fixture
def hipster(tmp_path):
    hipster = Hipster(output_folder=tmp_path, title="HipsterTest", max_order=0)
    return hipster


def test_generate_hips(hipster, model, tmp_path):
    hipster.generate_hips(model)

    assert filecmp.cmp(
        tmp_path / "HipsterTest/model/index.html",
        "tests/data/hipster/ref1/HipsterTest/model/index.html",
    )


def test_generate_catalog(hipster, model, tmp_path):
    datamodule = ShapesDataModule("tests/data/shapes", num_workers=1)
    hipster.generate_catalog(model, datamodule)

    assert filecmp.cmp(
        tmp_path / "HipsterTest/catalog.csv",
        "tests/data/hipster/ref1/HipsterTest/catalog.csv",
    )


def test_contains_equal_element():
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 2]

    contains_equal_element = any(x in list1 for x in list2)

    assert contains_equal_element is True


def test_find_best_rotation(model):
    torch.manual_seed(0)
    batch = torch.randn(2, 3, 4, 4)

    (
        best_scaled_image,
        best_rotations,
        best_coordinates,
        best_recon,
    ) = model.find_best_rotation(batch)

    assert best_scaled_image.shape == torch.Size([2, 3, 128, 128])
