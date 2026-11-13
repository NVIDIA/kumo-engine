# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched request-boundary probe; not an FM payload or model benchmark.

Run with an installed extension from this checkout:
    python diskgraph-pybind/python/tests/bench_columnar_requests.py --output result.json

Uses a fresh synthetic index, one native thread, warm caches, batch 512, and
alternating path order. Ingestion, lookup precomputation and exact Arrow-output
checks are outside timing. All paths consume the same fused Arrow event stream.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import diskgraph
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def fixture(root):
    users, per_user = 2048, 16
    pq.write_table(pa.table({"id": np.arange(users, dtype=np.int64)}), root / "users.parquet")
    pq.write_table(pa.table({
        "id": np.arange(users * per_user, dtype=np.int64),
        "owner": np.repeat(np.arange(users, dtype=np.int64), per_user),
        "amount": np.arange(users * per_user, dtype=np.float64) / 4,
    }), root / "events.parquet")
    tables = []
    for name, columns in [("users", [("id", "int64")]),
                          ("events", [("id", "int64"), ("owner", "int64"), ("amount", "float64")])]:
        table = {"name": name, "primary_key": "id",
                 "columns": [{"name": col, "ingested_dtype": dtype} for col, dtype in columns],
                 "source": {"type": "parquet", "paths": [str(root / f"{name}.parquet")]}}
        if name == "events":
            table["fkeys"] = [{"src_column": "owner", "dst_table": "users", "reverse": {}}]
        tables.append(table)
    diskgraph.ingest(json.dumps({"dataset_name": "columnar_probe", "num_shards": 1,
                                "tables": tables}), str(root / "index"), col_stats=False)
    return diskgraph.SamplerEngine(str(root / "index"), random_seed=41, num_threads=1)


def consume(events, verify):
    batches = defaultdict(list)
    rows = 0
    for event in events:
        if event["type"] == "stats":
            continue
        rows += event["batch"].num_rows
        if verify:
            batches[event["type"], event.get("rel_id", event.get("type_id"))].append(event["batch"])
    digest = hashlib.sha256()
    for key, parts in sorted(batches.items()):
        table = pa.Table.from_batches(parts).combine_chunks()
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        digest.update(str(key).encode())
        digest.update(sink.getvalue())
    return rows, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", default=[32000, 400000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="columnar-probe-") as tmp:
        engine = fixture(Path(tmp))
        relation = next(r["name"] for r in json.loads(engine.get_type_mapping())["edge_relations"]
                        if r["src_type"] == "users")
        plan = json.dumps({"layers": [{"fanouts": {relation: {
            "fanout": 16, "mode": "deterministic_uniform", "absent_key_policy": "uniform_fallback",
        }}, "default_fanout": None}]})
        paths = ["lookup_lists", "lookup_arrays", "lookup_fused", "materialized_lists", "materialized_arrays"]
        for count in args.anchors:
            keys = pa.array(np.arange(count, dtype=np.int64) % 2048)
            ids = engine.lookup_node_ids("users", keys, output_format="model")
            cutoffs = pa.array(np.full(count, 1700000000, dtype=np.int64))
            def run(path, verify=False):
                result = []
                for start in range(0, count, 512):
                    chunk = keys.slice(start, 512)
                    times = cutoffs.slice(start, 512)
                    options = dict(fused=True, seed_timestamp_unit=diskgraph.TimeUnit.Seconds)
                    if path == "lookup_fused":
                        events = engine.sample_from_keys({"users": chunk}, plan,
                                                         seed_timestamps={"users": times}, **options)
                    else:
                        roots = (engine.lookup_node_ids("users", chunk, output_format=(
                            "model" if path == "lookup_arrays" else "inspectable"))
                                 if path in {"lookup_lists", "lookup_arrays"} else ids.slice(start, 512))
                        if path == "materialized_lists":
                            roots = roots.to_pylist()
                        if path not in {"materialized_arrays", "lookup_arrays"}:
                            times = times.to_pylist()
                        events = engine.sample({"users": roots}, plan,
                                               seed_timestamps={"users": times}, **options)
                    result.append(consume(events, verify))
                return result
            # Every batch on every path is checked, including per-anchor ownership.
            expected = run(paths[0], verify=True)
            for path in paths[1:]:
                assert run(path, verify=True) == expected, path
            for path in paths:
                run(path)  # untimed warmup
            samples = {path: [] for path in paths}
            for repeat in range(args.repeats):
                for path in paths if repeat % 2 == 0 else paths[::-1]:
                    start = time.perf_counter()
                    run(path)
                    samples[path].append(time.perf_counter() - start)
            results.append({"anchors": count, "batch_size": 512, "exact_output_parity": True,
                            "seconds": samples,
                            "medians": {p: statistics.median(v) for p, v in samples.items()}})
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
