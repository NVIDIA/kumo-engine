# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Arrow tables written as ordinary connector source files."""
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _write_table(table, path):
    if not path.exists() or not pq.read_table(path).equals(table):
        pq.write_table(table, path)


def write_fixture(directory, *, users=32):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_table(pa.table({"user_id": pa.array(range(users), pa.int64())}),
                   directory / "users.parquet")
    rows = []
    for user in range(users):
        months = (5, 6, 9) if user % 2 == 0 else (5, 9)
        for month in months:
            rows.append({"order_id": len(rows), "user_id": user,
                         "order_time": datetime(2025, month, 15, tzinfo=timezone.utc)})
    schema = pa.schema([("order_id", pa.int64()), ("user_id", pa.int64()),
                        ("order_time", pa.timestamp("us", tz="UTC"))])
    _write_table(pa.Table.from_pylist(rows, schema=schema), directory / "orders.parquet")
    return directory
