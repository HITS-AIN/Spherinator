import filecmp
import shutil

import pytest

from hipster import Hipster
from models import RotationalVariationalAutoencoderPower


@pytest.fixture
def model():
    model = RotationalVariationalAutoencoderPower(z_dim=3)
    return model


def test_generate_hips(model):
    shutil.rmtree("/tmp/hipster")
    hipster = Hipster(output_folder="/tmp/hipster", title="HipsterTest", max_order=0)
    hipster.generate_hips(model)

    assert filecmp.cmp(
        "/tmp/hipster/HipsterTest/model/index.html",
        "tests/data/hipster/ref1/HipsterTest/model/index.html",
    )
