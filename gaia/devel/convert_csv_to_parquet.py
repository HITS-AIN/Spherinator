#!/usr/bin/env python3

"""Converts a CSV file to a Parquet file."""

import argparse
import sys

import pyarrow as pa
from pyarrow import csv, parquet


def skip_comment(row):
    if row.text.startswith("#"):
        return "skip"
    else:
        return "error"


def convert(in_path, out_path):
    """Converts a CSV file to a Parquet file."""
    writer = None
    parse_options = csv.ParseOptions(invalid_row_handler=skip_comment)
    with csv.open_csv(in_path, parse_options=parse_options) as reader:
        for next_chunk in reader:
            if next_chunk is None:
                break
            if writer is None:
                writer = parquet.ParquetWriter(out_path, next_chunk.schema)
            next_table = pa.Table.from_batches([next_chunk])
            writer.write_table(next_table)

    if writer is not None:
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Converts a CSV file to a Parquet file."
    )
    parser.add_argument(
        "in_path",
        help="The location of the input CSV data.",
    )
    parser.add_argument(
        "out_path",
        help="The location of the output Parquet data.",
    )

    args = parser.parse_args()
    convert(args.in_path, args.out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
