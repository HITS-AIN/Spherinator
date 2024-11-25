import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gaiaxpy import convert
from tqdm import tqdm

# DATA_DIR = "/hits/fast/ain/GAIA/cdn.gea.esac.esa.int/Gaia/gdr3/Spectroscopy/xp_continuous_mean_spectrum/"
DATA_DIR = "/local_data/doserbd/data/gaia"

sampling = np.linspace(0.0, 60.0, 40)

for file in tqdm(os.listdir(DATA_DIR)):
    if not file.endswith(".csv"):
        continue

    # df = pd.read_csv(DATA_DIR + file, comment='#', compression='gzip')
    # print(df)
    converted_data, sampling = convert(
        os.path.join(DATA_DIR, file), sampling=sampling, save_file=False
    )

    sum = converted_data["flux"].apply(lambda x: x.sum())
    converted_data["flux"] /= sum
    converted_data["flux_error"] /= sum

    # Use pyarrow to write the data to a parquet file
    table = pa.Table.from_pandas(converted_data)
    pq.write_table(table, f"{Path(file).stem}.parquet")
