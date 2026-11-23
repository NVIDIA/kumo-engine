# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare shared row-query range seeking against an unmodified clock baseline.

Run with the same Python environment and separate baseline/candidate package
directories. Preparation, source-ID checks and hashing are outside query timing.
Each pair uses fresh workers, balanced AB/BA order and one available CPU.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


CASES = ("historical_rows", "historical_lengths", "historical_sum",
         "recent_latest128", "unbounded", "mismatched")


def prepare(root, degree, clock):
    import diskgraph
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True)
    ids = np.arange(4 * degree, dtype=np.int64)
    unit = "ns" if clock == "nanoseconds" else "s"
    origin = 1_700_000_003 * (10**9 if unit == "ns" else 1)
    times = pa.array(origin + ids % degree, type=pa.timestamp(unit))
    pq.write_table(pa.table({"id": np.arange(4, dtype=np.int64)}), root / "users.parquet")
    pq.write_table(pa.table({"id": ids, "owner": ids // degree, "time": times,
                            "reference_time": times, "amount": ids % degree}),
                   root / "events.parquet")
    config = {"dataset_name": "shared_ranges", "timestamp_unit": clock,
              "num_shards": 1, "tables": [
        {"name": "users", "primary_key": "id", "columns": [
            {"name": "id", "ingested_dtype": "int64"}],
         "source": {"type": "parquet", "paths": [str(root / "users.parquet")]}},
        {"name": "events", "primary_key": "id", "columns": [
            {"name": name, "ingested_dtype": dtype} for name, dtype in
            [("id", "int64"), ("owner", "int64"), ("time", "timestamp"),
             ("reference_time", "timestamp"), ("amount", "int64")]],
         "source": {"type": "parquet", "paths": [str(root / "events.parquet")]},
         "fkeys": [{"src_column": "owner", "dst_table": "users", "reverse": {
             "time": {"table": "events", "column": "time", "unit": "minute",
                      "bitwidth": 32, "anchor_time": 1_700_000_003}}}]},
    ]}
    diskgraph.ingest(json.dumps(config), str(root / "index"), col_stats=False,
                     context_threads=1, encode_threads=1, pk_map_threads=1,
                     edges_map_threads=1, index_threads=1)
    (root / "shape.json").write_text(json.dumps({"degree": degree, "clock": clock,
                                                "origin": origin}))


def worker(root, case, repeats):
    import diskgraph
    import pyarrow as pa

    shape = json.loads((root / "shape.json").read_text())
    degree, clock, origin = (shape[key] for key in ("degree", "clock", "origin"))
    engine = diskgraph.SamplerEngine(str(root / "index"), num_threads=1)
    relation = next(r["name"] for r in json.loads(engine.get_type_mapping())["edge_relations"]
                    if r["src_type"] == "users")
    roots = {"users": engine.lookup_node_ids("users", pa.array(range(4)))}
    eligible = degree if case in ("recent_latest128", "unbounded") else max(1, degree // 100)
    limit = 128 if case == "recent_latest128" else None
    plan = {"frames": [{"name": "f", "root_type": "users",
        "path": [{"relation": relation, "destination_type": "events"}],
        "time_column": "reference_time" if case == "mismatched" else "time",
        "start_seconds": None, "stop_seconds": None if case == "unbounded" else 0,
        "cutoff_unit": clock, "window_unit": clock,
        "distinct": True, "columns": ["id", "amount"]}],
        "sequences": [{"name": "s", "frame": "f", "columns": ["id"],
                       "order": "descending", "limit": limit}], "aggregates": []}
    if case == "historical_sum":
        # Keep a shared sequence to exercise the general shared row frame,
        # rather than the separate specialized numeric PQL path.
        plan["aggregates"] = [{"name": "sum", "frame": "f", "operation": "sum",
                               "column": "amount"}]
    cutoffs = {"users": [origin + eligible - 1] * 4}
    query = engine.query_temporal_lengths if case == "historical_lengths" else engine.query_temporal
    plan_json = json.dumps(plan)

    def validate(result):
        count = min(eligible, limit) if limit else eligible
        sequence = result["sequences"][0]
        if case == "historical_lengths":
            actual = sequence["lengths"].to_pylist()
            assert actual == [count] * 4
        else:
            actual = sequence["batch"]["id"].to_pylist()
            assert sequence["offsets"] == [i * count for i in range(5)]
            assert actual == [owner * degree + i for owner in range(4)
                              for i in range(eligible - 1, eligible - count - 1, -1)]
            if case == "historical_sum":
                assert result["aggregates"][0]["values"] == [eligible * (eligible - 1) / 2] * 4
                actual += result["aggregates"][0]["values"]
        return hashlib.sha256(json.dumps(actual).encode()).hexdigest()

    for _ in range(2):
        result = query(roots, cutoffs, plan_json)
    digest = validate(result)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = query(roots, cutoffs, plan_json)
        samples.append(time.perf_counter_ns() - start)
        assert validate(result) == digest
    return {"median_ns": statistics.median(samples), "samples_ns": samples,
            "sha256": digest, "stats": result["stats"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--degrees", type=int, nargs="+", default=[256, 131072])
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--prepare", choices=["seconds", "nanoseconds"])
    parser.add_argument("--worker", choices=CASES)
    args = parser.parse_args()
    if args.cycles < 1 or args.repeats < 1 or any(d < 1 for d in args.degrees):
        parser.error("degrees, cycles and repeats must be positive")
    if args.prepare:
        prepare(args.output.resolve(), args.degrees[0], args.prepare)
        return
    if args.worker:
        print(json.dumps(worker(args.output.resolve(), args.worker, args.repeats)))
        return
    if args.baseline is None or args.candidate is None:
        parser.error("--baseline and --candidate are required")
    packages = {k: p.resolve() for k, p in
                [("baseline", args.baseline), ("candidate", args.candidate)]}
    hashes = {}
    for variant, package in packages.items():
        libraries = list((package / "diskgraph").glob("_c*.so"))
        if len(libraries) != 1:
            parser.error(f"expected one native library in {package}")
        hashes[variant] = hashlib.sha256(libraries[0].read_bytes()).hexdigest()
    if len(set(hashes.values())) != 2:
        parser.error("baseline and candidate libraries must differ")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"native_sha256": hashes, "cycles": args.cycles, "repeats": args.repeats,
               "cpu": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               "results": {}}

    def run(variant, *arguments):
        env = dict(os.environ, PYTHONPATH=str(packages[variant]), OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
        return subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                                        *map(str, arguments)], env=env, text=True)

    for degree in args.degrees:
        for clock in ("seconds", "nanoseconds"):
            fixture = output / f"{clock}-{degree}"
            run("baseline", fixture, "--prepare", clock, "--degrees", degree)
            for case in CASES:
                pairs = []
                for cycle in range(args.cycles):
                    order = ("baseline", "candidate") if cycle % 2 == 0 else ("candidate", "baseline")
                    pair = {variant: json.loads(run(variant, fixture, "--worker", case,
                                                   "--repeats", args.repeats)) for variant in order}
                    assert pair["baseline"]["sha256"] == pair["candidate"]["sha256"]
                    pairs.append(pair)
                ratios = [p["baseline"]["median_ns"] / p["candidate"]["median_ns"] for p in pairs]
                key = f"{clock}/{degree}/{case}"
                summary["results"][key] = {"pairs": pairs, "median_paired_speedup": statistics.median(ratios),
                    "ratio_range": [min(ratios), max(ratios)],
                    "source_sha256": hashlib.sha256((fixture / "events.parquet").read_bytes()).hexdigest()}
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(key, summary["results"][key]["median_paired_speedup"], flush=True)


if __name__ == "__main__":
    main()
