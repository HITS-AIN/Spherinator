import os
import numpy

from PIL import Image

import torch

from data.illustris_sdss_data_module import IllustrisSdssDataModule

if __name__ == "__main__":
    datamodel = IllustrisSdssDataModule(
        data_directories=["/local_data/AIN/SKIRT_synthetic_images/TNG100/sdss/snapnum_099/data/",
                          "/local_data/AIN/SKIRT_synthetic_images/TNG100/sdss/snapnum_095/data/",
                          "/local_data/AIN/SKIRT_synthetic_images/TNG50/sdss/snapnum_099/data/",
                          "/local_data/AIN/SKIRT_synthetic_images/TNG50/sdss/snapnum_095/data/",
                          "/local_data/AIN/SKIRT_synthetic_images/Illustris/sdss/snapnum_135/data/",
                          "/local_data/AIN/SKIRT_synthetic_images/Illustris/sdss/snapnum_131/data/"],
        extension=".fits",
        shuffle=False,
        minsize=100,
        batch_size=16,
        num_workers=16)
    datamodel.setup("val")

    for batch in datamodel.val_dataloader():
        for data, filename in zip(batch['image'], batch['filename']):
            image = torch.swapaxes(data, 0, 2)
            image = Image.fromarray((numpy.clip(image.numpy(),0,1)*255).astype(numpy.uint8) , mode="RGB")
            image.save(os.path.join(filename+".jpg"))
            print(filename+".jpg")
