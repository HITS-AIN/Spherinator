import os

import pandas as pd
import pyarrow as pa
from astropy.io import fits
from sympy import series

data_directories = [
    # "/hits/fast/its/doserbd/data/machine-learning/SKIRT_synthetic_images/Illustris/sdss/snapnum_135/data/",
    "/hits/fast/its/doserbd/data/machine-learning/two-images/TNG100/sdss/snapnum_099/data/",
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

            series.append(
                pd.Series(
                    {
                        "data": pa.Tensor.from_numpy(fits.getdata(filename, 0)),
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
