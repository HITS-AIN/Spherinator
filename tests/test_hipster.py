import filecmp

import pytest

from hipster import Hipster
from models import RotationalVariationalAutoencoderPower


@pytest.fixture
def model():
    model = RotationalVariationalAutoencoderPower(z_dim=3)
    return model


def test_generate_hips(model, tmp_path):
    hipster = Hipster(output_folder=tmp_path, title="HipsterTest", max_order=0)
    hipster.generate_hips(model)

    assert filecmp.cmp(
        tmp_path / "HipsterTest/model/index.html",
        "tests/data/hipster/ref1/HipsterTest/model/index.html",
    )


def test_contains_equal_element():
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 2]

    contains_equal_element = any(x in list1 for x in list2)

    assert contains_equal_element is True
