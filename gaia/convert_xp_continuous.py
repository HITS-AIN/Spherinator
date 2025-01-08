#!/usr/bin/env python3

"""Converts a CSV file to a Parquet file."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet
from tqdm import tqdm

list_of_arrays = [
    "bp_coefficients",
    "bp_coefficient_errors",
    "bp_coefficient_correlations",
    "rp_coefficients",
    "rp_coefficient_errors",
    "rp_coefficient_correlations",
]


def convert_to_parquet(path):
    for file in tqdm(os.listdir(path)):
        suffix = ".csv.gz"
        if not file.endswith(suffix):
            print(f"Skipping {file} as it is not a {suffix} file.")
            continue

        data = pd.read_csv(os.path.join(path, file), comment="#", compression="gzip")

        # Convert string entries to numpy arrays
        for array in list_of_arrays:
            data[array] = data[array].apply(
                lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=",")
            )

        # Use pyarrow to write the data to a parquet file
        table = pa.Table.from_pandas(data)

        parquet.write_table(
            table, f"{str(file).removesuffix(suffix)}.parquet", compression="snappy"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create parquet dataset with GAIA XP continuous data."
    )
    parser.add_argument(
        "path",
        help="Path to the directory containing the GAIA XP continuous data (csv files).",
    )

    args = parser.parse_args()
    convert_to_parquet(args.path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
