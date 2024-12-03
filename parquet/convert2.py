import io
import os

import numpy as np
import pandas as pd
from PIL import Image

from spherinator.data import IllustrisSdssDataModule

data_directories = [
    "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/fits/TNG100/sdss/snapnum_099/data",
    "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/fits/TNG100/sdss/snapnum_095/data",
    "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/fits/TNG50/sdss/snapnum_099/data",
    "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/fits/TNG50/sdss/snapnum_095/data",
    "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/fits/Illustris/sdss/snapnum_135/data",
    "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/fits/Illustris/sdss/snapnum_131/data",
    # "/hits/fast/its/doserbd/data/machine-learning/two-images/TNG100/sdss/snapnum_099/data/",
]
output_dir = (
    "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/parquet"
)
os.makedirs(output_dir, exist_ok=True)

datamodule = IllustrisSdssDataModule(
    data_directories=data_directories,
    extension="fits",
    minsize=100,
    batch_size=128,
    shuffle=False,
    num_workers=4,
)
datamodule.setup("processing")

dataloader = datamodule.processing_dataloader()
for n, (data, metadata) in enumerate(dataloader):
    series = []
    for i, image in enumerate(data):
        image = image.cpu().detach().numpy().T
        image = np.clip(image, 0, 1) * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        byte_image = io.BytesIO()
        image.save(byte_image, "jpeg")

        series.append(
            {
                "simulation": metadata["simulation"][i],
                "snapshot": metadata["snapshot"][i],
                "subhalo_id": metadata["subhalo_id"][i],
                "data": byte_image.getvalue(),
            }
        )
    df = pd.DataFrame(series)

    df.to_parquet(
        os.path.join(output_dir, f"{n}.parquet"),
        engine="pyarrow",
        compression="zstd",
    )
