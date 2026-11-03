# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: python arrow_offset_query_bench.py FRESH_INDEX_ROOT LABEL [model|inspectable]
# Reuse the generated index across baseline/candidate runs. Use a separate
# installed wheel for each label; the label only identifies the JSON result.

from pathlib import Path
import json, time, sys, hashlib, statistics
import diskgraph, pyarrow as pa, pyarrow.parquet as pq

root = Path(sys.argv[1])
variant = sys.argv[2]
output_format = sys.argv[3] if len(sys.argv) > 3 else "model"
root.mkdir(exist_ok=True)
if not (root / "index").exists():
    pq.write_table(pa.table({"id": list(range(100))}), root / "users.parquet")
    n = 10000
    pq.write_table(
        pa.table(
            {
                "id": list(range(n)),
                "owner": [i % 100 for i in range(n)],
                "time": pa.array(
                    [1700000000 + i // 100 for i in range(n)], type=pa.timestamp("s")
                ),
                "text": ["x" * 64 if i % 17 else None for i in range(n)],
                "value": [float(i) for i in range(n)],
                "blob": [b"abcdefgh" if i % 17 else None for i in range(n)],
                "tags": [
                    ["a", "bb", "ccc", "dddd"] if i % 17 else None for i in range(n)
                ],
            }
        ),
        root / "orders.parquet",
    )
    cfg = {
        "dataset_name": "offset_perf",
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
                    {"name": k, "ingested_dtype": v}
                    for k, v in [
                        ("id", "int64"),
                        ("owner", "int64"),
                        ("time", "timestamp"),
                        ("text", "string"),
                        ("value", "float64"),
                        ("blob", "bytes"),
                        ("tags", "string_array"),
                    ]
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
    diskgraph.ingest(
        json.dumps(cfg),
        str(root / "index"),
        col_stats=False,
        context_threads=1,
        encode_threads=1,
        pk_map_threads=1,
        edges_map_threads=1,
        index_threads=1,
    )
e = diskgraph.SamplerEngine(str(root / "index"))
relation = next(
    x["name"]
    for x in json.loads(e.get_type_mapping())["edge_relations"]
    if x["src_type"] == "users"
)
plan = {
    "frames": [
        {
            "name": "f",
            "root_type": "users",
            "path": [{"relation": relation, "destination_type": "orders"}],
            "time_column": "time",
            "start_seconds": -102,
            "stop_seconds": 0,
            "distinct": True,
            "columns": ["id", "text", "value", "blob", "tags"],
        }
    ],
    "sequences": [
        {
            "name": "s",
            "frame": "f",
            "columns": ["id", "text", "value", "blob", "tags"],
            "order": "descending",
            "limit": 32,
        }
    ],
    "aggregates": [],
}
ids = e.lookup_node_ids("users", pa.array([i % 100 for i in range(256)]))
times = [1700000101] * 256
p = json.dumps(plan)


def run():
    return e.query_temporal(
        {"users": ids}, {"users": times}, p, output_format=output_format
    )


first = run()


# Preserve a value-based output digest separately from timing.
def normalize(x):
    if isinstance(x, (pa.RecordBatch, pa.Table, pa.Array, pa.ChunkedArray)):
        return x.to_pylist()
    if isinstance(x, dict):
        return {
            k: normalize(v) for k, v in x.items() if k not in ("timings", "timings_ns")
        }
    if isinstance(x, (tuple, list)):
        return [normalize(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


digest = hashlib.sha256(
    json.dumps(normalize(first), sort_keys=True, default=str).encode()
).hexdigest()
del first
for _ in range(5):
    run()
start = time.perf_counter()
latencies = []
for _ in range(100):
    t0 = time.perf_counter()
    run()
    latencies.append(time.perf_counter() - t0)
print(
    json.dumps(
        {
            "variant": variant,
            "format": output_format,
            "seconds": time.perf_counter() - start,
            "queries": 100,
            "median_query_seconds": statistics.median(latencies),
            "p95_query_seconds": sorted(latencies)[94],
            "digest": digest,
        }
    )
)
