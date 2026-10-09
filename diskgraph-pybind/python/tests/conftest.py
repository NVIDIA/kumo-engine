# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from diskgraph._c import DiskgraphConfigError, DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError

import os
import json
import platform
import subprocess
from decimal import Decimal
import pyarrow as pa
import pyarrow.parquet as pq


def _portable_target_feature_flags():
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "-C target-feature=+aes,+sse2"
    if machine in {"aarch64", "arm64"}:
        return "-C target-feature=+aes,+neon"
    return ""


def _append_portable_rustflags(env):
    if env.get("GITHUB_ACTIONS"):
        return env

    target_feature_flags = _portable_target_feature_flags()
    if not target_feature_flags:
        return env

    rustflags = env.get("RUSTFLAGS", "").strip()
    env["RUSTFLAGS"] = (rustflags + " " + target_feature_flags).strip()
    return env


@pytest.fixture(scope="session")
def orders_graph(tmp_path_factory):
    """
    Generate three small tables for string-key, temporal, and two-hop tests.
    All rows are made up here; this fixture reads no external dataset.
    """
    root = tmp_path_factory.mktemp("synthetic_orders")
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    # 1. Create mock Parquet data
    users_schema = pa.schema([
        ("user_id", pa.string()),
        ("age", pa.int64())
    ])
    users_table = pa.Table.from_arrays(
        [pa.array(["u1", "u2", "u3"]), pa.array([25, 30, 35])],
        schema=users_schema
    )
    users_path = str(data_dir / "users.parquet")
    pq.write_table(users_table, users_path)

    orders_schema = pa.schema([
        ("order_id", pa.string()),
        ("user_id", pa.string()),
        ("merchant_id", pa.string()),
        ("timestamp", pa.int64())
    ])
    orders_table = pa.Table.from_arrays([
        pa.array(["o1", "o2", "o3"]),
        pa.array(["u1", "u2", "u1"]),
        pa.array(["m1", "m2", "m1"]),
        pa.array([1609459200000000, 1609545600000000, 1609632000000000])
    ], schema=orders_schema)
    orders_path = str(data_dir / "orders.parquet")
    pq.write_table(orders_table, orders_path)

    merchants_schema = pa.schema([("merchant_id", pa.string())])
    merchants_table = pa.Table.from_arrays([pa.array(["m1", "m2"])], schema=merchants_schema)
    merchants_path = str(data_dir / "merchants.parquet")
    pq.write_table(merchants_table, merchants_path)

    # 2. Link orders to users and merchants, with reverse edges and timestamps
    config = {
        "dataset_name": "synthetic_orders",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "age", "ingested_dtype": "int64"}
                ],
                "primary_key": "user_id",
                "source": {"type": "parquet", "paths": [users_path]}
            },
            {
                "name": "merchants",
                "columns": [{"name": "merchant_id", "ingested_dtype": "string"}],
                "primary_key": "merchant_id",
                "source": {"type": "parquet", "paths": [merchants_path]}
            },
            {
                "name": "orders_parquet",
                "columns": [
                    {"name": "order_id", "ingested_dtype": "string"},
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "merchant_id", "ingested_dtype": "string"},
                    {"name": "timestamp", "ingested_dtype": "timestamp"}
                ],
                "primary_key": "order_id",
                "create_time": "timestamp",
                "fkeys": [
                    {"src_column": "user_id", "dst_table": "users", "reverse": {}},
                    {"src_column": "merchant_id", "dst_table": "merchants", "reverse": {}}
                ],
                "source": {"type": "parquet", "paths": [orders_path]}
            }
        ]
    }

    config_path = root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # 3. Build the Index using cargo run (assumes we are in the workspace)
    try:
        # Find cargo project root (assuming tests are run from diskgraph-pybind or root)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        print(f"\nBuilding mock index at {index_dir} using cargo run...")
        subprocess.run(
            ["cargo", "run", "--release", "--bin", "ingest", "--",
             "--source", str(config_path),
             "--dest", "local",
             "--output", str(index_dir)],
            cwd=project_root,
            check=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Index ingestion failed with exit code {e.returncode}")

    return str(index_dir)


@pytest.fixture(scope="session")
def list_features_graph(tmp_path_factory):
    """
    Graph with intlist (int64_array) and stringlist (string_array) feature
    columns. Used to verify that the pybind wheel can ingest and serve
    these column types without crashing.

    Schema:
      items: item_id (int64 PK), tag_ids (List<Int64>), tags (List<Utf8>)
    """
    root = tmp_path_factory.mktemp("list_features")
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    # Build Parquet with list columns
    import pyarrow as pa
    import pyarrow.parquet as pq

    item_ids = pa.array([0, 1, 2, 3], type=pa.int64())
    tag_ids = pa.array(
        [[10, 20, 30], [40], [], [1, 2, 3, 4, 5]],
        type=pa.list_(pa.int64()),
    )
    tags = pa.array(
        [["sports", "sale"], ["kids"], [], ["pro", "audio"]],
        type=pa.list_(pa.utf8()),
    )

    table = pa.table({"item_id": item_ids, "tag_ids": tag_ids, "tags": tags})
    items_path = str(data_dir / "items.parquet")
    pq.write_table(table, items_path)

    config = {
        "dataset_name": "list_features_test",
        "num_shards": 1,
        "tables": [
            {
                "name": "items",
                "columns": [
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "tag_ids", "ingested_dtype": "int64_array"},
                    {"name": "tags", "ingested_dtype": "string_array"},
                ],
                "primary_key": "item_id",
                "source": {"type": "parquet", "paths": [items_path]},
            }
        ],
    }

    config_path = root / "config.json"
    import json
    with open(config_path, "w") as f:
        json.dump(config, f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    env = _append_portable_rustflags(dict(os.environ))
    try:
        subprocess.run(
            ["cargo", "run", "--release", "--bin", "ingest", "--",
             "--source", str(config_path),
             "--dest",   "local",
             "--output", str(index_dir)],
            cwd=project_root,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"list_features ingest failed with exit code {e.returncode}")

    return str(index_dir)


@pytest.fixture(scope="session")
def typed_primary_key_graph(tmp_path_factory):
    """Graph covering every Python representation used by PK catalog scans."""
    root = tmp_path_factory.mktemp("typed_primary_keys")
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    table_specs = [
        ("signed_ids", pa.array([-5, 7, 7], type=pa.int64()),
         {"ingested_dtype": "int64"}),
        ("unsigned_ids", pa.array([7, 2**63 + 1], type=pa.uint64()),
         {"ingested_dtype": "u_int64"}),
        ("string_ids", pa.array(["  a ", "0xAB"]),
         {"ingested_dtype": "string"}),
        ("bytes_ids", pa.array([b"\x00\xff", b"raw"], type=pa.binary()),
         {"ingested_dtype": "bytes"}),
        ("uuid_ids",
         pa.array([
             "00000000-0000-0000-0000-000000000001",
             "00000000-0000-0000-0000-000000000002",
         ]), {"ingested_dtype": "uuid"}),
        ("uuid256_ids",
         pa.array(["01" * 32, "02" * 32]),
         {"ingested_dtype": "uuid256"}),
        ("ip_ids", pa.array(["192.0.2.1", "2001:db8::1"]),
         {"ingested_dtype": "ip"}),
        ("decimal128_ids",
         pa.array([Decimal("123.00"), Decimal("7.00"), Decimal("7.00")],
                  type=pa.decimal128(38, 2)),
         {"ingested_dtype": "decimal128", "precision": 38, "scale": 2}),
        ("decimal256_ids",
         pa.array([Decimal("123.00"), Decimal("7.00"), Decimal("7.00")],
                  type=pa.decimal256(76, 2)),
         {"ingested_dtype": "decimal256", "precision": 76, "scale": 2}),
    ]
    tables = []
    for name, values, column_config in table_specs:
        path = str(data_dir / f"{name}.parquet")
        pq.write_table(pa.table({"id": values}), path)
        tables.append({
            "name": name,
            "columns": [{"name": "id", **column_config}],
            "primary_key": "id",
            "source": {"type": "parquet", "paths": [path]},
        })

    config_path = root / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "dataset_name": "typed_primary_keys_test",
            "num_shards": 1,
            "tables": tables,
        }, f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    env = _append_portable_rustflags(dict(os.environ))
    try:
        subprocess.run(
            ["cargo", "run", "--release", "--bin", "ingest", "--",
             "--source", str(config_path),
             "--dest", "local",
             "--output", str(index_dir)],
            cwd=project_root,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"typed primary-key ingest failed with exit code {e.returncode}")

    return str(index_dir)


@pytest.fixture(scope="session")
def list_features_graph_with_inner_nulls(tmp_path_factory):
    """
    Graph whose list feature columns contain inner null elements and
    legitimate empty strings. Used to verify the drop-null contract:
      * Null elements are dropped from the persisted on-disk list.
      * Empty string elements are preserved (the null bit is the only signal).

    Schema:
      items: item_id (int64 PK), tag_ids (List<Int64>), tags (List<Utf8>)

    Source data (with nulls marked as N and empty string as ""):
      item 0 → tag_ids=[1, N, 3], tags=["a", N, "b"]
      item 1 → tag_ids=[N, N],    tags=[N, N]
      item 2 → tag_ids=[0, N, 7], tags=["x", "", "y"]
      item 3 → tag_ids=[10, 20],  tags=["p", "", N, "q"]

    Expected decoded data:
      item 0 → tag_ids=[1, 3],     tags=["a", "b"]
      item 1 → tag_ids=[],         tags=[]
      item 2 → tag_ids=[0, 7],     tags=["x", "", "y"]
      item 3 → tag_ids=[10, 20],   tags=["p", "", "q"]
    """
    root = tmp_path_factory.mktemp("list_features_inner_nulls")
    data_dir = root / "data"
    data_dir.mkdir()
    index_dir = root / "index"

    item_ids = pa.array([0, 1, 2, 3], type=pa.int64())
    tag_ids = pa.array(
        [[1, None, 3], [None, None], [0, None, 7], [10, 20]],
        type=pa.list_(pa.int64()),
    )
    tags = pa.array(
        [["a", None, "b"], [None, None], ["x", "", "y"], ["p", "", None, "q"]],
        type=pa.list_(pa.utf8()),
    )

    table = pa.table({"item_id": item_ids, "tag_ids": tag_ids, "tags": tags})
    items_path = str(data_dir / "items.parquet")
    pq.write_table(table, items_path)

    config = {
        "dataset_name": "list_features_inner_nulls_test",
        "num_shards": 1,
        "tables": [
            {
                "name": "items",
                "columns": [
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "tag_ids", "ingested_dtype": "int64_array"},
                    {"name": "tags", "ingested_dtype": "string_array"},
                ],
                "primary_key": "item_id",
                "source": {"type": "parquet", "paths": [items_path]},
            }
        ],
    }

    config_path = root / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    env = _append_portable_rustflags(dict(os.environ))
    try:
        subprocess.run(
            ["cargo", "run", "--release", "--bin", "ingest", "--",
             "--source", str(config_path),
             "--dest",   "local",
             "--output", str(index_dir)],
            cwd=project_root,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"list_features_with_inner_nulls ingest failed with exit code {e.returncode}")

    return str(index_dir)


@pytest.fixture
def parent_index(tmp_path):
    """Ingest a tiny users+orders parent DG index. Returns index path."""
    import diskgraph

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parent_root = tmp_path / "parent_idx"

    n_users = 12
    orders_per_user = 6
    pq.write_table(
        pa.table({"user_id": pa.array(list(range(n_users)), type=pa.int64())}),
        data_dir / "users.parquet",
    )
    order_ids, order_uids, order_times = [], [], []
    time_buckets = [500, 1500, 2500, 3500, 4500]
    for u in range(n_users):
        for k in range(orders_per_user):
            order_ids.append(u * orders_per_user + k)
            order_uids.append(u)
            order_times.append(time_buckets[k % len(time_buckets)])
    pq.write_table(
        pa.table({
            "order_id": pa.array(order_ids, type=pa.int64()),
            "user_id": pa.array(order_uids, type=pa.int64()),
            "order_time": pa.array(order_times, type=pa.timestamp("s")),
        }),
        data_dir / "orders.parquet",
    )
    diskgraph.ingest(
        config_json=json.dumps({
            "dataset_name": "ttng_stage1_parent",
            "num_shards": 1,
            "tables": [
                {
                    "name": "users",
                    "columns": [{"name": "user_id", "ingested_dtype": "int64"}],
                    "primary_key": "user_id",
                    "source": {"type": "parquet", "paths": [str(data_dir / "users.parquet")]},
                },
                {
                    "name": "orders",
                    "columns": [
                        {"name": "order_id", "ingested_dtype": "int64"},
                        {"name": "user_id", "ingested_dtype": "int64"},
                        {"name": "order_time", "ingested_dtype": "timestamp"},
                    ],
                    "primary_key": "order_id",
                    "create_time": "order_time",
                    "fkeys": [{
                        "src_column": "user_id",
                        "dst_table": "users",
                        "forward": {},
                        "reverse": {
                            "time": {
                                "table": "orders",
                                "column": "order_time",
                                "bitwidth": 32,
                                "unit": "s",
                            },
                        },
                    }],
                    "source": {"type": "parquet", "paths": [str(data_dir / "orders.parquet")]},
                },
            ],
        }),
        output_dir=str(parent_root),
        col_stats=False,
    )
    return parent_root


@pytest.fixture
def node_prediction_plan():
    """Return PQL plan JSON for binary-classification node prediction on users."""
    from diskgraph import DiskGraphTTBuilder, EdgeRef, SplitSpec

    builder = DiskGraphTTBuilder()
    builder.temporal([1500, 2500, 3500], forecast_length=1, unit="seconds")
    builder.node_prediction()
    builder.for_each("users", "user_id")
    builder.set_target_stype("numerical")
    builder.aggregate(
        "cnt",
        EdgeRef("users", "user_id", "orders", "user_id"),
        "count",
    )
    builder.split_ratio(SplitSpec(
        kind="TimeRangeSplit",
        ranges=[(0, 1999), (2000, 2999), (3000, 5000)],
        is_temporal=True,
        timestamp_unit="seconds",
        offset_unit="seconds",
        unit="seconds",
    ))
    return builder.build()
