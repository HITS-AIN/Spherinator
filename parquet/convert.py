import os

import numpy as np
import pandas as pd
import pyarrow as pa
from astropy.io import fits

data_directories = [
    # "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/Illustris/sdss/snapnum_135/data/",
    "/hits/fast/its/doserbd/data/machine-learning/one-image/TNG100/sdss/snapnum_099/data/",
]
extension = "fits"
output_dir = "illustris_sdss"
os.makedirs(output_dir, exist_ok=True)

series = []
for data_directory in data_directories:
    for filename in sorted(os.listdir(data_directory)):
        if filename.endswith(extension):
            filename = os.path.join(data_directory, filename)
            splits = filename[: -(len(extension) + 1)].split("/")

            data = fits.getdata(filename, 0)
            data = np.array(data).astype(np.float32)

            series.append(
                pd.Series(
                    {
                        "data": pa.Tensor.from_numpy(data),
                        "simulation": splits[-5],
                        "snapshot": splits[-3].split("_")[1],
                        "subhalo_id": splits[-1].split("_")[1],
                    }
                )
            )

df = pd.DataFrame(series)

print(df)

i = 0
df.to_parquet(
    os.path.join(output_dir, f"{i}.parquet"),
    engine="pyarrow",
    compression="zstd",
)
