# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-only paired-build probe. Prepare once, run each build on the same index.

Use separate processes/PYTHONPATHs for baseline and candidate. Output hashes must
match before interpreting timings. Preparation and hashing are outside timings.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import diskgraph
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def prepare(root, users, degree, unit):
    count = users * degree
    ids = np.arange(count, dtype=np.int64)
    times = 1_700_000_000 + ids % degree
    write_fixture(root, ids // degree, times, unit,
                  {"users": users, "degree": degree, "unit": unit})


def write_fixture(root, owners, times, unit, shape):
    root.mkdir(parents=True, exist_ok=False)
    users = shape["users"]
    ids = np.arange(len(times), dtype=np.int64)
    pq.write_table(pa.table({"id": np.arange(users, dtype=np.int64)}), root / "users.parquet")
    columns = {"id": ids, "owner": owners,
               "time": pa.array(times, type=pa.timestamp("s")),
               "reference_time": pa.array(times, type=pa.timestamp("s")),
               "amount": ids.astype(np.float64) / 8,
               "payload": pa.array([("event-%d:" % i) + "x" * 128 for i in ids])}
    for i in range(8):
        columns[f"value_{i}"] = ids.astype(np.float64) + i
    pq.write_table(pa.table(columns), root / "events.parquet")
    specs = [("id", "int64"), ("owner", "int64"), ("time", "timestamp"),
             ("reference_time", "timestamp"), ("amount", "float64"), ("payload", "string")]
    specs += [(f"value_{i}", "float64") for i in range(8)]
    config = {"dataset_name": "coarse_benchmark", "num_shards": 2, "tables": [
        {"name": "users", "primary_key": "id", "columns": [{"name": "id", "ingested_dtype": "int64"}],
         "source": {"type": "parquet", "paths": [str(root / "users.parquet")]}},
        {"name": "events", "primary_key": "id",
         "columns": [{"name": n, "ingested_dtype": t} for n, t in specs],
         "source": {"type": "parquet", "paths": [str(root / "events.parquet")]},
         "fkeys": [{"src_column": "owner", "dst_table": "users", "reverse": {
             "time": {"table": "events", "column": "time", "unit": unit,
                      "bitwidth": 32, "anchor_time": 1_700_000_000}}}]},
    ]}
    (root / "config.json").write_text(json.dumps(config))
    diskgraph.ingest(json.dumps(config), str(root / "index"), col_stats=False,
                     context_threads=2, encode_threads=2, pk_map_threads=2,
                     edges_map_threads=2, index_threads=2)
    (root / "shape.json").write_text(json.dumps(shape))


def digest(result):
    h = hashlib.sha256()
    for seq in result["sequences"]:
        for name, value in sorted(seq.items()):
            h.update(name.encode())
            if isinstance(value, pa.RecordBatch):
                h.update(value.serialize().to_pybytes())
            elif isinstance(value, pa.Array):
                h.update(json.dumps(value.to_pylist()).encode())
            else:
                h.update(json.dumps(value).encode())
    h.update(json.dumps(result.get("aggregates", [])).encode())
    return h.hexdigest()


def benchmark(root, label, repeats):
    shape = json.loads((root / "shape.json").read_text())
    engine = diskgraph.SamplerEngine(str(root / "index"), num_threads=1)
    reverse = next(r["name"] for r in json.loads(engine.get_type_mapping())["edge_relations"]
                   if r["src_type"] == "users")
    roots = {"users": engine.lookup_node_ids("users", pa.array(np.arange(shape["users"], dtype=np.int64)))}
    cutoffs = {"users": [1_700_000_000 + shape["degree"] // 2 + 17] * shape["users"]}
    columns = ["id", "amount", "payload"] + [f"value_{i}" for i in range(8)]
    cases = {}
    for name, lower, upper, limit, aggregate, clock in [
        ("narrow_sequence", -300, 0, 64, False, "time"),
        ("narrow_shared", -300, 0, 64, True, "time"),
        ("narrow_all_rows", -300, 0, None, False, "time"),
        ("broad_sequence", -100_000, 100_000, 64, False, "time"),
        ("unbounded_control", None, None, 64, False, "time"),
        ("mismatched_control", -300, 0, 64, False, "reference_time"),
    ]:
        plan = {"frames": [{"name": "f", "root_type": "users",
                            "path": [{"relation": reverse, "destination_type": "events"}],
                            "time_column": clock, "start_seconds": lower, "stop_seconds": upper,
                            "distinct": True, "columns": columns}],
                "sequences": [{"name": "s", "frame": "f", "columns": columns,
                               "order": "descending", "limit": limit}],
                "aggregates": ([{"name": "sum", "frame": "f", "operation": "sum", "column": "amount"}]
                               if aggregate else [])}
        cases[name] = (engine.query_temporal, json.dumps(plan))
        if name in ("narrow_sequence", "broad_sequence", "mismatched_control"):
            cases[name + "_lengths"] = (engine.query_temporal_lengths, json.dumps(plan))
    output = {"label": label, "shape": shape, "platform": platform.platform(),
              "python": platform.python_version(), "diskgraph": diskgraph.__file__, "cases": {}}
    for name, (query, plan) in cases.items():
        for _ in range(2):
            result = query(roots, cutoffs, plan)
        expected = digest(result)
        calls, native, phases = [], [], []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = query(roots, cutoffs, plan)
            calls.append(time.perf_counter_ns() - start)
            assert digest(result) == expected
            timings = result["timings_ns"]
            native.append(timings["total"])
            phases.append(timings)
        output["cases"][name] = {"sha256": expected, "stats": result["stats"],
                                  "call_ns": calls, "native_ns": native, "phases_ns": phases,
                                  "median_call_ns": statistics.median(calls),
                                  "median_native_ns": statistics.median(native)}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--users", type=int, default=16)
    parser.add_argument("--degree", type=int, default=8192)
    parser.add_argument("--unit", default="minute", choices=["s", "minute", "hour", "day"])
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root.resolve(), args.users, args.degree, args.unit)
    else:
        result = benchmark(args.root.resolve(), args.label, args.repeats)
        payload = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(payload)
        else:
            print(payload)
