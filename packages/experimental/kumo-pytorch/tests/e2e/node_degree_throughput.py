# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare matching native DiskGraph batches with and without point-in-time degree."""

import argparse
import json
import math
import statistics
import tempfile
from bisect import bisect_right
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns

import kumo.settings

from kumo.pytorch import api
from tests.e2e.scored_walk_throughput import _open_task
from tests.test_scored_walks import star, write_parquet


def fixture(args, directory):
    schema, _ = star()
    nboards = max(2, args.pins // 16)
    rows = {
        "pins": [{"id": pin} for pin in range(args.pins)],
        "boards": [{"id": board} for board in range(nboards + 1)],
        "membership": [],
    }
    timestamps = defaultdict(list)
    for pin in range(args.pins):
        for board, at in (
            (1 + pin % nboards, 1),
            (1 + (pin + 1) % nboards, 1 + pin % 16),
            (0, 1 + pin % 8),
        ):
            rows["membership"].append(dict(id=len(rows["membership"]), pin=pin, board=board, at=at))
            timestamps["pins", pin].append(at)
            timestamps["boards", board].append(at)
    for times in timestamps.values():
        times.sort()
    requests = tuple(
        api.SampleRequest(str(i), (i * 131) % args.pins, 8 if i % 2 else 16)
        for i in range(args.batch_size * args.batches)
    )
    dataset, anchors = write_parquet(schema, rows, requests, directory)
    return schema, dataset, anchors, {r.sample_id: r.as_of_time for r in requests}, timestamps


def contract(schema, args, *, degree):
    layers = (
        api.GraphLayer(
            (api.WalkSample("walk", ("membership",), {"pins": 25, "boards": 10}, 0.2, 1024),)
        ),
    )
    relations = {"pins": "pin_links", "boards": "board_links"}
    values = (
        {
            table: {
                "degree": api.Aggregate(
                    api.Rows((relation,), api.TimeWindow("at", stop=0)), "count"
                )
            }
            for table, relation in relations.items()
        }
        if degree
        else {}
    )
    graph = api.NeighborhoodSpec(layers, node_values=values)
    fields = ("id", "degree") if degree else ("id",)
    rep = api.Graph(
        {
            table: api.Fields(tuple(api.Encode(name, api.Raw()) for name in fields))
            for table in relations
        }
    )
    inputs = {"query": api.Input(api.NeighborhoodSource("graph"), rep, None)}
    sets = {}
    if args.positives:
        sets["positive"] = api.CandidateSetSpec(
            api.EntitySpec("pins"),
            neighborhoods={"graph": graph},
            draw=api.DrawNeighbors("graph", ("walk.pins",), count=args.positives, weight="score"),
        )
        inputs["positive"] = api.Input(
            api.NeighborhoodSource("graph", candidates="positive"), rep, None
        )
    return api.DataLoaderSpec(
        schema,
        api.SampleSpec(api.EntitySpec("pins"), neighborhoods={"graph": graph}, candidate_sets=sets),
        api.BatchSpec(inputs=inputs, batching=api.Batching(max_samples=args.batch_size)),
    )


def check_batches(baseline, enriched, cutoffs, timestamps):
    """Read every degree against source Parquet data outside the timed loop."""
    assert enriched.metadata["sample_ids"] == baseline.metadata["sample_ids"]
    groups = enriched.metadata.get("candidate_entity_ids", {})
    assert groups == baseline.metadata.get("candidate_entity_ids", {})
    for key in ("walk_pushes", "walk_decoded_entries"):
        assert enriched.metadata[key] == baseline.metadata[key]
    parent_times = [cutoffs[s] for s in enriched.metadata["sample_ids"]]
    n = 0
    for name, graph in enriched.inputs.items():
        base = baseline.inputs[name]
        times = (
            parent_times
            if name == "query"
            else [at for at, group in zip(parent_times, groups[name], strict=True) for _ in group]
        )
        for table in ("pins", "boards"):
            if isinstance(graph, api.ColumnarGraph):
                node, control = graph.nodes[table], base.nodes[table]
                assert node.ids.equals(control.ids) and node.owners.equals(control.owners)
                ids, owners = node.ids.to_pylist(), node.owners.to_pylist()
                degrees = node.features["degree"].to_pylist()
            else:
                node, control = graph["nodes"][table], base["nodes"][table]
                assert node["ids"].equal(control["ids"]) and node["batch"].equal(control["batch"])
                ids, owners = node["ids"].tolist(), node["batch"].tolist()
                degrees = node["features"]["degree"]["values"].tolist()
            assert degrees == [
                bisect_right(timestamps[table, node_id], times[owner])
                for node_id, owner in zip(ids, owners, strict=True)
            ]
            n += len(ids)
    return n


def timed_pass(task, args):
    elapsed = []
    start = perf_counter_ns()
    batches = iter(task.batches(output_format=args.output_format))
    for _ in range(args.batches):
        next(batches)
        now = perf_counter_ns()
        elapsed.append(now - start)
        start = now
    assert next(batches, None) is None
    return elapsed


def summary(ns, args, occurrences):
    seconds = sum(ns) / 1e9
    return {
        "batches": len(ns),
        "p50_ms": statistics.median(ns) / 1e6,
        "p95_ms": sorted(ns)[math.ceil(len(ns) * 0.95) - 1] / 1e6,
        "anchors_per_second": len(ns) * args.batch_size / seconds,
        "node_occurrences_per_second": occurrences * (len(ns) / args.batches) / seconds,
    }


def measure(args):
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        folder = Path(tmp)
        kumo.settings.configure(scratch_dir=folder / "scratch")
        schema, ds, anchors, cutoffs, timestamps = fixture(args, folder)
        tasks = {
            label: stack.enter_context(
                _open_task(
                    contract(schema, args, degree=flag),
                    ds,
                    api.DiskGraphEngine(num_threads=1),
                    anchors,
                )
            )
            for label, flag in (("baseline", False), ("degree", True))
        }
        verified = sum(
            check_batches(left, right, cutoffs, timestamps)
            for left, right in zip(
                *(t.batches(output_format=args.output_format) for t in tasks.values()), strict=True
            )
        )
        samples = {name: [] for name in tasks}
        for repeat in range(args.repeats):
            for name in tasks if repeat % 2 == 0 else reversed(tasks):
                samples[name].extend(timed_pass(tasks[name], args))
    return {
        **vars(args),
        "node_occurrences_per_pass": verified,
        **{name: summary(times, args, verified) for name, times in samples.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-format", choices=("columnar", "model"), default="model")
    parser.add_argument("--positives", type=int, default=0)
    parser.add_argument("--pins", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if min(args.pins, args.batch_size, args.batches, args.repeats) < 1 or args.positives < 0:
        parser.error("sizes must be positive and positives must be nonnegative")
    print(json.dumps(measure(args), sort_keys=True))
