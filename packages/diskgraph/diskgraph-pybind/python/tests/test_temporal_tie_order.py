# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tie breaks must order values like `(type(v).__name__, repr(v))`."""

import itertools
import json

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq

TIME = 1_700_000_000
FLOATS = [
    1e16, 1e-5, 1.5e-05, 123.0, -0.0, 0.0, float("nan"), float("inf"),
    float("-inf"), 1e300, 5e-324, 0.0001, 9999999999999998.0,
]
STRINGS = [
    "a", "it's", 'say "hi"', "both ' and \"", "back\\slash", "tab\t", "nl\n",
    "\x7f", "\x80", "\xa0", "é", " ", "​", "　", "😀", "",
]


def python_key(value):
    return type(value).__name__, repr(value)


def query(root, orders, dtypes, users, frame_columns, sequences, aggregates):
    pq.write_table(pa.table({"id": list(range(users))}), root / "users.parquet")
    pq.write_table(orders, root / "orders.parquet")
    config = {
        "dataset_name": "tie_order",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "primary_key": "id",
                "columns": [{"name": "id", "ingested_dtype": "int64"}],
                "source": {"type": "parquet", "paths": [str(root / "users.parquet")]},
            },
            {
                "name": "orders",
                "primary_key": "id",
                "create_time": "time",
                "columns": [
                    {"name": name, "ingested_dtype": dtype} for name, dtype in dtypes
                ],
                "fkeys": [
                    {
                        "src_column": "owner",
                        "dst_table": "users",
                        "reverse": {
                            "time": {"table": "orders", "column": "time", "unit": "s"}
                        },
                    }
                ],
                "source": {"type": "parquet", "paths": [str(root / "orders.parquet")]},
            },
        ],
    }
    diskgraph.ingest(json.dumps(config), str(root / "index"), col_stats=False)
    engine = diskgraph.SamplerEngine(str(root / "index"))
    relation = next(
        r["name"]
        for r in json.loads(engine.get_type_mapping())["edge_relations"]
        if r["src_type"] == "users"
    )
    plan = {
        "frames": [
            {
                "name": "f",
                "root_type": "users",
                "path": [{"relation": relation, "destination_type": "orders"}],
                "time_column": "time",
                "distinct": True,
                "columns": frame_columns,
            }
        ],
        "sequences": sequences,
        "aggregates": aggregates,
    }
    ids = engine.lookup_node_ids("users", pa.array(list(range(users))))
    return engine.query_temporal(
        {"users": ids}, {"users": [TIME + 1] * users}, json.dumps(plan)
    )


def test_string_primary_key_ties_follow_python_repr(tmp_path):
    n = len(STRINGS)
    orders = pa.table(
        {
            "id": STRINGS,
            "owner": [0] * n,
            "time": pa.array([TIME] * n, type=pa.timestamp("s")),
        }
    )
    result = query(
        tmp_path,
        orders,
        [("id", "string"), ("owner", "int64"), ("time", "timestamp")],
        1,
        ["id"],
        [{"name": "s", "frame": "f", "columns": ["id"], "order": "ascending", "limit": None}],
        [],
    )
    actual = result["sequences"][0]["batch"].column("id").to_pylist()
    assert actual == sorted(STRINGS, key=python_key)


def test_float_mode_ties_follow_python_repr(tmp_path):
    # Each root holds one distinct pair; mode breaks the count tie by the smaller key.
    pairs = list(itertools.combinations(FLOATS, 2))
    values = [value for pair in pairs for value in pair]
    orders = pa.table(
        {
            "id": list(range(len(values))),
            "owner": [index // 2 for index in range(len(values))],
            "value": pa.array(values, type=pa.float64()),
            "time": pa.array([TIME] * len(values), type=pa.timestamp("s")),
        }
    )
    result = query(
        tmp_path,
        orders,
        [("id", "int64"), ("owner", "int64"), ("value", "float64"), ("time", "timestamp")],
        len(pairs),
        ["value"],
        [],
        [{"name": "m", "frame": "f", "operation": "mode", "column": "value"}],
    )
    actual = [repr(value) for value in result["aggregates"][0]["values"]]
    assert actual == [repr(min(pair, key=python_key)) for pair in pairs]
