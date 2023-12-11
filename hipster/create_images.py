import os
from pathlib import Path

import numpy
import torch
from PIL import Image

from data.spherinator_data_module import SpherinatorDataModule


class Mixin:
    def create_images(self, datamodule: SpherinatorDataModule):
        datamodule.setup("images")

        for i, image in enumerate(datamodule.dataloader_images()):
            image = torch.swapaxes(image, 0, 2)
            image = Image.fromarray(
                (numpy.clip(image.numpy(), 0, 1) * 255).astype(numpy.uint8), mode="RGB"
            )
            metadata = datamodule.data_images.dataset.get_metadata[i]
            filename = os.path.join(
                self.output_folder,
                self.title,
                "thumbnails",
                metadata["simulation"],
                metadata["snapshot"],
                metadata["subhalo_id"] + ".jpg",
            )
            image.save(filename)
