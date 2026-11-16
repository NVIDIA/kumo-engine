#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Converts a CSV file to parquet, automatically performing type inference on
# the columns.
#
# Usage:
#
#   ./tools/dataset/convert_csv_to_parquet.py <CSV_S3_PATH> <OUTPUT_S3_PATH>
#
# CSV_S3_PATH points to a single CSV file. Must end with ".csv"
#
# OUTPUT_S3_PATH is the name of the destination parquet file.
#
# To convert multiple tables at once, please use the following script:
#
#   ./tools/dataset/convert_many_csv_to_parquet.sh
#

import argparse
import os

from io_utils import fsutil
from kumo.connector.csv_connector import CSVConnector


def run(csv_s3_path: str, output_s3_path: str) -> None:
    # Parse CSV filename
    root_dir, fname = os.path.split(csv_s3_path)
    table_name, ext = os.path.splitext(fname)
    if ext != ".csv":
        print(f"CSV_S3_PATH must end with `.csv` but got: {csv_s3_path}")
        exit(1)

    # Validate S3 file does not exist
    if fsutil.exists(output_s3_path):
        print(f"A file already exists at the output path. "
              f"Please delete: {output_s3_path}")
        exit(1)

    # Read CSV file and infer metadata
    conn = CSVConnector("name", root_dir)
    meta = conn.get_table_metadata([table_name])
    cols = meta[0].all_cols()

    print(f"Reading csv table '{table_name}' from '{root_dir}'")
    df = conn.read_table(table_name, cols=cols)
    print("Got the following dataframe:")
    print(df)
    print(df.dtypes)

    # Write to S3
    print(f"Writing parquet to '{output_s3_path}'")
    df.to_parquet(output_s3_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "CSV_S3_PATH",
        help="a single CSV file",
    )
    parser.add_argument(
        "OUTPUT_S3_PATH",
        help="the output parquet file name",
    )
    args = parser.parse_args()
    run(args.CSV_S3_PATH, args.OUTPUT_S3_PATH)
