import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet
from tqdm import tqdm


def convert_to_parquet(path):
    for file in tqdm(os.listdir(path)):
        if not file.endswith(".csv"):
            continue

        # data = pyarrow.csv.read_csv(path + file)
        # data = pd.read_csv(os.path.join(path, file), comment="#", compression="gzip")
        data = pd.read_csv(os.path.join(path, file), comment="#")

        # Convert string entries to numpy arrays
        data["flux"] = data["flux"].apply(
            lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=",")
        )
        data["flux_error"] = data["flux_error"].apply(
            lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=",")
        )

        # Normalize the flux and flux_error columns
        sum = data["flux"].apply(lambda x: x.sum())
        data["flux"] /= sum
        data["flux_error"] /= sum

        # Use pyarrow to write the data to a parquet file
        table = pa.Table.from_pandas(data)
        parquet.write_table(table, f"{Path(file).stem}.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create parquet dataset with GAIA xp sampled data."
    )
    parser.add_argument(
        "path",
        help="Path to the directory containing the GAIA xp sampled data (csv files).",
    )

    args = parser.parse_args()
    convert_to_parquet(args.path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
