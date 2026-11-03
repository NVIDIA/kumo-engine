# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native temporal query scenarios motivated by FM extraction, not production data.

Run baseline and candidate in separate processes on identical prepared indexes.
The source oracle validates selected IDs and lengths outside the timed region.
This compares PR82 to its coarse-clock optimization, not sample() to query_temporal.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import diskgraph
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_coarse_filter import digest, write_fixture


DAY = 86400
EPOCH = 1_700_000_000
SPAN = 730 * DAY
DEGREES = (16, 128, 1024, 8192)
PATTERNS = ("steady", "recent", "dormant", "burst")


def prepare(root, unit):
    rng = np.random.default_rng(82)
    owners, clocks, cohorts = [], [], []
    for degree in DEGREES:
        for pattern in PATTERNS:
            for replicate in range(2):
                owner = len(cohorts)
                if pattern == "steady":
                    offsets = rng.choice(SPAN, degree, replace=False)
                elif pattern == "dormant":
                    offsets = rng.choice(SPAN - 180 * DAY, degree, replace=False)
                else:
                    recent_span = (90 if pattern == "recent" else 7) * DAY
                    recent_count = degree * 3 // 4
                    offsets = np.concatenate([
                        rng.choice(SPAN - recent_span, degree - recent_count, replace=False),
                        SPAN - recent_span + rng.choice(recent_span, recent_count, replace=False),
                    ])
                clocks.extend((EPOCH + np.sort(offsets)).tolist())
                owners.extend([owner] * degree)
                cohorts.append({"owner": owner, "degree": degree, "pattern": pattern,
                                "replicate": replicate})
    shape = {"profile": "fm_windows_v1", "seed": 82, "users": len(cohorts),
             "rows": len(clocks), "history_days": 730, "unit": unit, "cohorts": cohorts,
             "synthetic": True}
    write_fixture(root, np.array(owners, dtype=np.int64), np.array(clocks, dtype=np.int64),
                  unit, shape)


def scenarios(shape):
    """A core sweep plus focused fanout, density, activity, and control cases."""
    all_roots = list(range(shape["users"]))

    def case(name, *, age=1, window=30, width="narrow", limit=24,
             api="sequence", roots=None, clock="time", upper=0):
        return {"name": name, "cutoff_age_days": age, "window_days": window,
                "width": width, "limit": limit, "api": api,
                "roots": all_roots if roots is None else roots, "clock": clock, "upper": upper}

    for age in (1, 365):
        for window in (None, 7, 30, 90, 365):
            tag = "upper_only" if window is None else f"w{window}"
            for width in ("narrow", "wide"):
                yield case(f"age{age}_{tag}_{width}_k24", age=age, window=window, width=width)
            yield case(f"age{age}_{tag}_lengths", age=age, window=window, api="lengths")
    for window in (None, 7, 30, 90, 365):
        tag = "upper_only" if window is None else f"w{window}"
        yield case(f"age1_{tag}_narrow_k64", window=window, limit=64)
    for degree in DEGREES:
        roots = [c["owner"] for c in shape["cohorts"] if c["degree"] == degree]
        for window in (None, 30):
            tag = "upper_only" if window is None else "w30"
            yield case(f"degree{degree}_{tag}", roots=roots, window=window)
    for pattern in PATTERNS:
        roots = [c["owner"] for c in shape["cohorts"] if c["pattern"] == pattern]
        for window in (None, 30):
            tag = "upper_only" if window is None else "w30"
            yield case(f"pattern_{pattern}_{tag}", roots=roots, window=window)
    yield case("age1_w30_shared_sum", api="shared")
    yield case("age1_w30_all_rows", limit=None)
    yield case("unbounded_control", window=None, upper=None)
    yield case("mismatched_control", clock="reference_time")


def oracle(source, spec):
    cutoff = EPOCH + SPAN - spec["cutoff_age_days"] * DAY + spec.get("cutoff_offset_seconds", 0)
    lower = None if spec["window_days"] is None else cutoff - spec["window_days"] * DAY
    upper = None if spec["upper"] is None else cutoff + spec["upper"]
    selected, counts, without_lower, reached = [], [], [], 0
    total_amount = 0.0
    for owner in spec["roots"]:
        ids, times = source[owner]
        reached += len(ids)
        mask = np.ones(len(ids), dtype=bool) if upper is None else times <= upper
        before_lower = ids[mask][::-1]
        if lower is not None:
            mask &= times > lower
        eligible = ids[mask][::-1]
        counts.append(len(eligible))
        total_amount += float(eligible.sum()) / 8
        selected.append(eligible[:spec["limit"]].tolist())
        without_lower.append(before_lower[:spec["limit"]].tolist())
    return {"selected": selected, "eligible_per_root": counts, "reached": reached,
            "eligible": sum(counts), "returned": sum(map(len, selected)),
            "roots_changed_by_lower_bound": sum(a != b for a, b in zip(selected, without_lower)),
            "sum_amount": total_amount}


def validate(result, expected, spec):
    seq = result["sequences"][0]
    if spec["api"] == "lengths":
        assert seq["lengths"].to_pylist() == list(map(len, expected["selected"])), (
            spec, seq["lengths"].to_pylist(), list(map(len, expected["selected"])))
    else:
        assert seq["batch"].column("id").to_pylist() == [i for rows in expected["selected"] for i in rows]
        assert np.diff(seq["offsets"]).tolist() == list(map(len, expected["selected"]))


def benchmark(root, label, repeats, case_names=None, cutoff_offset=0):
    shape = json.loads((root / "shape.json").read_text())
    table = pq.read_table(root / "events.parquet", columns=["id", "owner", "time"])
    ids = table["id"].to_numpy()
    owners = table["owner"].to_numpy()
    # Parquet stores timestamp[s] as timestamp[ms]; normalize before integer comparisons.
    times = table["time"].cast(pa.timestamp("s")).cast(pa.int64()).to_numpy()
    source = {i: (ids[owners == i], times[owners == i]) for i in range(shape["users"])}
    engine = diskgraph.SamplerEngine(str(root / "index"), num_threads=1)
    reverse = next(r["name"] for r in json.loads(engine.get_type_mapping())["edge_relations"]
                   if r["src_type"] == "users")
    output = {"label": label, "shape": shape, "platform": platform.platform(),
              "python": platform.python_version(), "diskgraph": diskgraph.__file__, "cases": {}}
    # Identical deterministic shuffle in each process avoids always timing windows in size order.
    specs = list(scenarios(shape))
    if case_names is not None:
        assert set(case_names) <= {spec["name"] for spec in specs}, case_names
        specs = [spec for spec in specs if spec["name"] in case_names]
    np.random.default_rng(82).shuffle(specs)
    for spec in specs:
        spec["cutoff_offset_seconds"] = cutoff_offset
        columns = ["id", "amount"]
        if spec["width"] == "wide":
            columns += ["payload"] + [f"value_{i}" for i in range(8)]
        lower = None if spec["window_days"] is None else -spec["window_days"] * DAY
        plan = {"frames": [{"name": "f", "root_type": "users",
                            "path": [{"relation": reverse, "destination_type": "events"}],
                            "time_column": spec["clock"], "start_seconds": lower,
                            "stop_seconds": spec["upper"], "distinct": True, "columns": columns}],
                "sequences": [{"name": "s", "frame": "f", "columns": columns,
                               "order": "descending", "limit": spec["limit"]}],
                "aggregates": ([{"name": "sum", "frame": "f", "operation": "sum", "column": "amount"}]
                               if spec["api"] == "shared" else [])}
        plan = json.dumps(plan)
        roots = {"users": engine.lookup_node_ids("users", pa.array(spec["roots"], type=pa.int64()))}
        cutoffs = {"users": [EPOCH + SPAN - spec["cutoff_age_days"] * DAY + cutoff_offset]
                  * len(spec["roots"])}
        query = engine.query_temporal_lengths if spec["api"] == "lengths" else engine.query_temporal
        expected = oracle(source, spec)
        for _ in range(2):
            result = query(roots, cutoffs, plan)
        validate(result, expected, spec)
        expected_digest = digest(result)
        calls, native, phases = [], [], []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = query(roots, cutoffs, plan)
            calls.append(time.perf_counter_ns() - start)
            assert digest(result) == expected_digest
            validate(result, expected, spec)
            native.append(result["timings_ns"]["total"])
            phases.append(result["timings_ns"])
        output["cases"][spec["name"]] = {
            "spec": spec, "selectivity": {k: v for k, v in expected.items() if k != "selected"},
            "sha256": expected_digest, "stats": result["stats"], "call_ns": calls,
            "native_ns": native, "phases_ns": phases,
            "median_call_ns": statistics.median(calls), "median_native_ns": statistics.median(native)}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--unit", default="hour", choices=["s", "minute", "hour", "day"])
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", dest="case_names")
    parser.add_argument("--cutoff-offset-seconds", type=int, default=0,
                        help="Shift cutoffs off the index's bucket boundaries")
    parser.add_argument("--serve", action="store_true",
                        help="Accept JSON case-name lists on stdin for closely paired runs")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root.resolve(), args.unit)
    elif args.serve:
        for line in sys.stdin:
            print(json.dumps(benchmark(args.root.resolve(), args.label, args.repeats,
                                       json.loads(line), args.cutoff_offset_seconds)), flush=True)
    else:
        result = json.dumps(benchmark(args.root.resolve(), args.label, args.repeats,
                                      args.case_names, args.cutoff_offset_seconds), indent=2) + "\n"
        if args.output:
            args.output.write_text(result)
        else:
            print(result)
