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
