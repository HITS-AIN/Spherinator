import os
import numpy

from PIL import Image

import torch

from data.illustris_sdss_data_module import IllustrisSdssDataModule

if __name__ == "__main__":
    datamodel = IllustrisSdssDataModule(
        data_directories=[
            "/local_data/AIN/SKIRT_synthetic_images/TNG100/sdss/snapnum_099/data/",
            "/local_data/AIN/SKIRT_synthetic_images/TNG100/sdss/snapnum_095/data/",
            "/local_data/AIN/SKIRT_synthetic_images/TNG50/sdss/snapnum_099/data/",
            "/local_data/AIN/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/",
            "/local_data/AIN/SKIRT_synthetic_images/Illustris/sdss/snapnum_135/data/",
            "/local_data/AIN/SKIRT_synthetic_images/Illustris/sdss/snapnum_131/data/",
        ],
        extension=".fits",
        shuffle=False,
        minsize=100,
        batch_size=1,
        num_workers=16,
    )
    datamodel.setup("images")
    datamodel.setup("thumbnail_images")

    print(
        datamodel.val_dataloader().dataset.total_files,
        len(datamodel.val_dataloader().dataset),
    )

    for item in datamodel.val_dataloader().dataset:
        image = torch.swapaxes(item["image"], 0, 2)
        image = Image.fromarray(
            (numpy.clip(image.numpy(), 0, 1) * 255).astype(numpy.uint8), mode="RGB"
        )
        filename = os.path.join(
            "/local_data/AIN/Data/HiPSter",
            "Illustris",
            "thumbnails",
            item["metadata"]["simulation"],
            item["metadata"]["snapshot"],
            item["metadata"]["subhalo_id"] + ".jpg",
        )
        print(filename)
        image.save(filename)
