import filecmp

import pytest
import torch

from data import ShapesDataModule
from hipster import Hipster
from models import RotationalVariationalAutoencoderPower


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
    datamodule = ShapesDataModule(
        "tests/data/shapes", exclude_files=["boxes.npy", "circles.npy", "triangles.npy"]
    )
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
    batch = torch.randn(2, 3, 4, 4)

    best_image, rot, coord, loss = model.find_best_rotation(batch)

    assert best_image.shape == torch.Size([2, 3, 128, 128])
    assert torch.allclose(rot, torch.Tensor([0, 0]), rtol=1e-3)

    assert coord.shape == torch.Size([2, 3])
    # assert torch.allclose(
    #     coord,
    #     torch.Tensor([[-0.4980, 0.4326, -0.7515], [-0.5011, 0.4362, -0.7474]]),
    #     rtol=1e-3,
    # )

    assert loss.shape == torch.Size([2])
    # assert torch.allclose(loss, torch.Tensor([0.2119, 0.2122]), rtol=1e-3)
