import os

import numpy
import torch
from PIL import Image


class Mixin:
    def create_images(self, datamodel):
        datamodel.setup("images")

        for i, image in enumerate(datamodel.image_dataloader()):
            image = torch.swapaxes(image, 0, 2)
            image = Image.fromarray(
                (numpy.clip(image.numpy(), 0, 1) * 255).astype(numpy.uint8), mode="RGB"
            )
            metadata = datamodel.data_images.dataset.get_metadata[i]
            filename = os.path.join(
                self.output_folder,
                self.title,
                "thumbnails",
                metadata["simulation"],
                metadata["snapshot"],
                metadata["subhalo_id"] + ".jpg",
            )
            image.save(filename)
