#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Uploads an S3 dataset to the given snowflake table name.
#
# Usage:
#
#   ./tools/dataset/upload_table_to_snowflake.py <S3_PATH> <DB>.<SCHEMA>.<TABLE_NAME> # noqa
#
# S3_PATH points to a single parquet file, or a directory containing multiple
# parquet files in the same dataset.
#
# DB is the name of the database in snowflake (must already exist).
#
# SCHEMA also must already exist.
#
# TABLE_NAME is destination table in snowflake. If it already exists, the
# script will fail (unless the --append flag is set)
#
# Use --account to override the default account. This script reads the secret
# named <account>.snowflakecomputing.com from the configured secret manager.
# The secret must contain the password and can specify the user (default: kumo).
#
# To upload multiple datasets at once, please use the following script:
#
#   ./tools/dataset/upload_many_tables_to_snowflake.sh
#

import argparse
import math
from dataclasses import dataclass

from io_utils import fsutil, parquet_util
from io_utils.secrets import get_secrets
from kumo.connector.snowflake_connector import SnowflakeConnector

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # type: ignore
    print(
        "INFO: tqdm not available. Install with 'pip install tqdm' for progress bars."
    )


@dataclass
class SnowflakePath:
    db: str
    schema: str
    table: str


def make_connector(database: str, schema: str, account: str,
                   warehouse: str) -> SnowflakeConnector:
    secrets = get_secrets(f"{account.lower()}.snowflakecomputing.com")
    username = secrets.get("user", "kumo")
    password = secrets.get("password")

    conn = SnowflakeConnector(name="fake_name", account=account.lower(),
                              warehouse=warehouse, database=database,
                              schema_name=schema)

    conn.open({"user": username, "password": password})
    return conn


def parse_snowflake_path(snowflake_path: str) -> SnowflakePath:
    parts = snowflake_path.split('.', maxsplit=2)
    if len(parts) != 3 or len(parts[0]) == 0 or len(parts[1]) == 0 or len(
            parts[2]) == 0:
        print(f"Snowflake path must be of the form <DB>.<SCHEMA>.<TABLE> "
              f"but got '{snowflake_path}'")
        exit(1)
    return SnowflakePath(parts[0], parts[1], parts[2])


def confirm_table_not_exists(connector: SnowflakeConnector, table: str,
                             append: bool) -> None:
    if connector.table_exists(table) and not append:
        raise ValueError(
            f"Snowflake table with name '{table}' already exists. "
            f"Please delete the table, or pass in the `--append` flag to "
            f"append to the existing table.")


def confirm_s3_exists(path: str) -> None:
    if not fsutil.exists(path):
        print(f"S3 path '{path}' does not exist.")
        exit(1)


def estimate_total_batches(s3_path: str,
                           target_bytes_per_batch: int = 2 << 20) -> int:
    """Estimate the total number of batches based on parquet metadata."""
    try:
        metadata = parquet_util.get_parquet_metadata(s3_path)

        if metadata.avg_row_size == 0:
            return 0

        # Max row count per batch
        rows_per_batch = min(
            math.ceil(target_bytes_per_batch / metadata.avg_row_size),
            128 << 10,  # default value (128Ki) for rows_per_batch
        )

        if rows_per_batch == 0:
            return 1

        # Estimate total batches
        total_batches = max(1, metadata.num_rows // rows_per_batch)
        return total_batches

    except Exception:
        # If we can't estimate, return 0 to disable progress bar
        return 0


def run(s3_path: str, snowflake_path: str, append: bool, account: str,
        warehouse: str) -> None:
    dest = parse_snowflake_path(snowflake_path)
    connector = make_connector(dest.db, dest.schema, account, warehouse)
    confirm_table_not_exists(connector, dest.table, append)
    confirm_s3_exists(s3_path)

    print(f"Downloading {s3_path}...")

    # Estimate total batches for progress bar
    total_batches = estimate_total_batches(s3_path) if TQDM_AVAILABLE else 0

    # Get the batches iterator
    batches = parquet_util.to_batches(s3_path)

    # Wrap with tqdm if available and we can estimate batches
    if TQDM_AVAILABLE and total_batches > 0:
        batches = tqdm(batches, desc="[Estimated] Uploading batches",
                       unit="batch", total=total_batches)
    elif TQDM_AVAILABLE:
        batches = tqdm(batches, desc="Uploading batches", unit="batch")

    for shard in batches:
        if not TQDM_AVAILABLE:
            print(f"Writing part {shard.batch_number}")

        connector.write_pandas(dest.table, shard.df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "S3_PATH",
        help=("a single parquet file, or a directory containing "
              "multiple parquet files in the same dataset."),
    )
    parser.add_argument(
        "SNOWFLAKE_PATH",
        help=("the <DB>.<SCHEMA>.<TABLE_NAME> where "
              "the table should be uploaded"),
    )
    parser.add_argument(
        '--append', action='store_true',
        help=('Append to the table, if it already '
              'exists in Snowflake.'))
    parser.add_argument(
        '--account',
        default="xva19026",
        help=('the account name of the snowflake account'),
    )
    parser.add_argument(
        '--warehouse',
        default="WH_XS",
        help=('the warehouse name of the snowflake account'),
    )
    args = parser.parse_args()
    run(args.S3_PATH, args.SNOWFLAKE_PATH, args.append, args.account,
        args.warehouse)
