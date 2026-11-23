# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example input data is ordinary typed Arrow/Parquet, without bundle tooling."""
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def test_example_parquet_fixture(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/local_sdm/fixture.py"
    spec = importlib.util.spec_from_file_location("example_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = module.write_fixture(tmp_path / "source", users=4)
    users = pq.read_table(directory / "users.parquet")
    orders = pq.read_table(directory / "orders.parquet")
    assert users["user_id"].to_pylist() == [0, 1, 2, 3]
    assert orders.schema.field("order_time").type == pa.timestamp("us", tz="UTC")
    assert orders["order_id"].to_pylist() == list(range(10))
    june = [row["user_id"] for row in orders.to_pylist() if row["order_time"].month == 6]
    assert june == [0, 2]
    assert sorted(p.name for p in directory.iterdir()) == ["orders.parquet", "users.parquet"]
    before = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
    module.write_fixture(directory, users=4)
    assert {path.name: path.stat().st_mtime_ns for path in directory.iterdir()} == before
