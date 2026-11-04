# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uncapped range-admission benchmark; paired by temporal_fm_windows_compare.py.

Eight independent anchors, configurable degree, 99% newer than the old cutoff.
The recent-cutoff case and a separate degree-256 fixture are ordinary controls.
Preparation, independent ID/length validation, and hashes are outside timings.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import diskgraph
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_coarse_filter import digest


ANCHOR = 1_700_000_003


def prepare(root, degree, unit):
    root.mkdir(parents=True, exist_ok=False)
    ids = np.arange(8 * degree, dtype=np.int64)
    pq.write_table(pa.table({"id": np.arange(8, dtype=np.int64)}), root / "users.parquet")
    pq.write_table(pa.table({"id": ids, "owner": ids // degree,
                            "time": pa.array(ANCHOR + ids % degree, type=pa.timestamp("s"))}),
                   root / "events.parquet")
    config = {"dataset_name": "indexed_range", "num_shards": 2, "tables": [
        {"name": "users", "primary_key": "id",
         "columns": [{"name": "id", "ingested_dtype": "int64"}],
         "source": {"type": "parquet", "paths": [str(root / "users.parquet")]}},
        {"name": "events", "primary_key": "id",
         "columns": [{"name": n, "ingested_dtype": t}
                     for n, t in [("id", "int64"), ("owner", "int64"), ("time", "timestamp")]],
         "source": {"type": "parquet", "paths": [str(root / "events.parquet")]},
         "fkeys": [{"src_column": "owner", "dst_table": "users", "reverse": {
             "time": {"table": "events", "column": "time", "unit": unit,
                      "bitwidth": 32, "anchor_time": ANCHOR}}}]},
    ]}
    diskgraph.ingest(json.dumps(config), str(root / "index"), col_stats=False,
                     context_threads=2, encode_threads=2, pk_map_threads=2,
                     edges_map_threads=2, index_threads=2)
    (root / "shape.json").write_text(json.dumps({"users": 8, "degree": degree, "unit": unit}))


def worker(args):
    root = args.root.resolve()
    shape = json.loads((root / "shape.json").read_text())
    degree = shape["degree"]
    engine = diskgraph.SamplerEngine(str(root / "index"), num_threads=1)
    relation = next(r["name"] for r in json.loads(engine.get_type_mapping())["edge_relations"]
                    if r["src_type"] == "users")
    roots = {"users": engine.lookup_node_ids("users", pa.array(range(8)))}
    for line in sys.stdin:
        output = {"label": args.label, "shape": shape, "cases": {}}
        for name in json.loads(line):
            if name not in {"old_all_rows", "old_latest128", "recent_latest128", "old_lengths"}:
                raise ValueError(name)
            eligible = degree if name == "recent_latest128" else degree // 100
            limit = 128 if name.endswith("latest128") else None
            cutoffs = {"users": [ANCHOR + eligible - 1] * 8}
            if args.cutoff_offset_seconds:
                raise ValueError("This workload fixes selectivity; use the FM worker for shifted cutoffs")
            plan = json.dumps({"frames": [{"name": "f", "root_type": "users",
                "path": [{"relation": relation, "destination_type": "events"}],
                "time_column": "time", "start_seconds": None, "stop_seconds": 0,
                "distinct": True, "columns": ["id"]}], "sequences": [{
                "name": "s", "frame": "f", "columns": ["id"], "order": "descending",
                "limit": limit}]})
            query = engine.query_temporal_lengths if name == "old_lengths" else engine.query_temporal
            for _ in range(2):
                result = query(roots, cutoffs, plan)
            count = min(eligible, limit) if limit else eligible
            if name == "old_lengths":
                assert result["sequences"][0]["lengths"].to_pylist() == [count] * 8
            else:
                sequence = result["sequences"][0]
                assert sequence["offsets"] == [i * count for i in range(9)]
                assert sequence["batch"].column("id").to_pylist() == [
                    owner * degree + i for owner in range(8)
                    for i in range(eligible - 1, eligible - count - 1, -1)]
            expected = digest(result)
            calls, native, phases = [], [], []
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                result = query(roots, cutoffs, plan)
                calls.append(time.perf_counter_ns() - started)
                assert digest(result) == expected
                native.append(result["timings_ns"]["total"])
                phases.append(result["timings_ns"])
            output["cases"][name] = {"sha256": expected, "stats": result["stats"],
                "call_ns": calls, "native_ns": native, "phases_ns": phases,
                "median_call_ns": statistics.median(calls),
                "median_native_ns": statistics.median(native)}
        print(json.dumps(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--degree", type=int, default=1_048_576)
    parser.add_argument("--unit", choices=["s", "minute", "hour", "day"], default="minute")
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--cutoff-offset-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.degree < 100 or args.repeats < 1:
        parser.error("degree must be >= 100 and repeats >= 1")
    if args.prepare:
        prepare(args.root.resolve(), args.degree, args.unit)
    else:
        worker(args)
