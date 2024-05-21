import math
from pathlib import Path

import numpy
import skimage.io as io
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image


def create_allsky(
    data_directory: Path,
    dir_id: int = 0,
    edge_width: int = 64,
    max_order: int = 4,
    extension: str = "jpg",
):
    for order in range(max_order + 1):
        width = math.floor(math.sqrt(12 * 4**order))
        height = math.ceil(12 * 4**order / width)
        result = torch.zeros((edge_width * height, edge_width * width, 3))

        for i in range(12 * 4**order):
            file = (
                data_directory
                / Path("Norder" + str(order))
                / Path("Dir" + str(dir_id))
                / Path("Npix" + str(i) + "." + extension)
            )
            if not file.exists():
                raise RuntimeError("File not found: " + str(file))

            image = torch.swapaxes(torch.Tensor(io.imread(file)), 0, 2) / 255.0
            image = transforms.functional.resize(image, [64, 64], antialias=True)
            image = torch.swapaxes(image, 0, 2)

            x = i % width
            y = math.floor(i / width)
            result[
                y * edge_width : (y + 1) * edge_width,
                x * edge_width : (x + 1) * edge_width,
            ] = image
        image = Image.fromarray(
            (numpy.clip(result.numpy(), 0, 1) * 255).astype(numpy.uint8), mode="RGB"
        )
        image.save(data_directory / Path("Norder" + str(order)) / Path("Allsky.jpg"))
