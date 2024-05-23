from pathlib import Path

import numpy as np
from PIL import Image

from spherinator.data import ImagesDataset


def test_suffix():
    filename = "test.jpg"
    extensions = ["jpg", "png"]
    assert Path(filename).suffix[1:] == "jpg"
    assert Path(filename).suffix[1:] in extensions


def test_load_image():
    image = Image.open("tests/data/images/Abra/1.jpg")
    assert image.size == (224, 224)
    assert image.mode == "RGB"
    assert image.format == "JPEG"
    array = np.asarray(image)
    assert array.shape == (224, 224, 3)


def test_dataset():
    dataset = ImagesDataset("tests/data/images")
    assert len(dataset) == 2
    data = dataset[0]
    assert data.shape == (3, 224, 224)
