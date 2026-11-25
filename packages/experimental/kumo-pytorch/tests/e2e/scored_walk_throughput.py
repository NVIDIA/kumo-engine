# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time anchor and caller-supplied candidate graphs on the same synthetic Pin dataset."""

import argparse
import json
import math
import resource
import statistics
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import kumo.settings

from kumo.pytorch import api
from tests.test_candidate_neighborhoods import parquet_inputs
from tests.test_scored_walks import star, write_parquet


def measure(args):
    schema, _ = star()
    # The first board is a hub. Each Pin additionally belongs to two local boards.
    board_count = max(2, args.pins // 16)
    rows = {
        "pins": [{"id": pin} for pin in range(args.pins)],
        "boards": [{"id": board} for board in range(board_count + 1)],
        "membership": [],
    }
    for pin in range(args.pins):
        memberships = [1 + pin % board_count, 1 + (pin + 1) % board_count]
        if pin % 4 == 0:
            memberships.append(0)
        for board in memberships:
            rows["membership"].append(dict(id=len(rows["membership"]), pin=pin, board=board, at=1))
    root_table, root_count, cutoff = "pins", args.pins, 1
    prefix = ()
    if args.history:
        root_table, root_count, cutoff = "users", (args.pins + 3) // 4, 4
        integer = api.DType("int64")
        clock = api.DType("timestamp", unit="s", timezone="UTC")
        schema = api.DataSchema(
            schema.tables
            + (
                api.TableSpec("users", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
                api.TableSpec(
                    "actions",
                    "id",
                    {"id": integer, "user": integer, "pin": integer, "at": clock},
                    time_column="at",
                ),
            ),
            schema.relations
            + (
                api.RelationSpec("user_actions", "users", "id", "actions", "user"),
                api.RelationSpec("action_pin", "pins", "id", "actions", "pin"),
            ),
        )
        rows["users"] = [{"id": user} for user in range(root_count)]
        rows["actions"] = [
            {"id": pin, "user": pin // 4, "pin": pin, "at": 1 + pin % 4} for pin in range(args.pins)
        ]
        prefix = (
            api.GraphLayer((api.GraphSample("user_actions", 4, strategy="recent"),)),
            api.GraphLayer((api.GraphSample("action_pin", 1, direction="reverse"),)),
        )
    requests = [
        api.SampleRequest(str(i), (i * 131) % root_count, cutoff)
        for i in range(args.batch_size * args.batches)
    ]
    if args.mode == "walk":
        layers = (
            api.GraphLayer(
                (
                    api.WalkSample(
                        "ranked", ("membership",), {"pins": 25, "boards": 10}, 0.2, args.budget
                    ),
                )
            ),
        )
    else:
        layers = (
            api.GraphLayer((api.GraphSample("pin_links", 25),)),
            api.GraphLayer((api.GraphSample("board_links", 1, direction="reverse"),)),
        )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec(root_table),
            neighborhoods={"graph": api.NeighborhoodSpec(prefix + layers)},
        ),
        api.BatchSpec(
            inputs={"graph": api.Input(api.NeighborhoodSource("graph"), api.Graph(), None)},
            batching=api.Batching(max_samples=args.batch_size),
        ),
    )
    if args.positives is not None:
        positive = api.CandidateSetSpec(
            api.EntitySpec("pins"), neighborhoods={"graph": api.NeighborhoodSpec(layers)}
        )
        candidate_input = api.Input(
            api.NeighborhoodSource("graph", candidates="positive"), api.Graph(), None
        )
        contract = replace(
            contract,
            sample=replace(contract.sample, candidate_sets={"positive": positive}),
            batch=replace(
                contract.batch, inputs={**contract.batch.inputs, "positive_graph": candidate_input}
            ),
        )
        requests = [
            replace(
                request,
                candidates={
                    "positive": tuple(
                        api.CandidateRequest((i * 131 + j + 1) % args.pins)
                        for j in range(args.positives)
                    )
                },
            )
            for i, request in enumerate(requests)
        ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        kumo.settings.configure(scratch_dir=root / "scratch")
        writer = write_parquet if args.positives is None else parquet_inputs
        dataset, anchors = writer(schema, rows, requests, root)
        refusal = RuntimeError("benchmark selected the Python fallback")
        with (
            api.prepare(contract, dataset, api.DiskGraphEngine(num_threads=args.threads)) as plan,
            plan.materialize(anchors) as task,
            patch.object(task._anchors, "_requests", side_effect=refusal),
            patch.object(task._assembler, "batches", side_effect=refusal),
        ):

            def batches():
                return task.batches(output_format=args.output_format, shuffle=False, seed=17)

            source = task._assembler._source
            first_pass = _measure_passes(batches, args, 1, source)
            warm = _measure_passes(batches, args, args.repeats, source)
    return {
        **{key: value for key, value in vars(args).items() if key != "json_path"},
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "first_pass": first_pass,
        **warm,
    }


def _edge_counts(graph):
    if isinstance(graph, api.ColumnarGraph):
        return {name: edge.index.num_rows for name, edge in graph.edges.items()}
    return {name: edge["index"].shape[1] for name, edge in graph["edges"].items()}


def _measure_passes(batches, args, repeats, source):
    latencies, startup = [], []
    returned, starts, frontier_starts, elapsed = 0, 0, 0, 0.0
    positive_count = positive_edges = 0
    sample_times, graph_times = [], []
    counters = {"walk_pushes": 0, "walk_decoded_entries": 0} if args.mode == "walk" else {}
    for _ in range(repeats):
        begun = perf_counter()
        iterator = iter(batches())
        elapsed += perf_counter() - begun
        for i in range(args.batches):
            start = perf_counter()
            batch = next(iterator)
            ready = perf_counter()
            latencies.append(ready - start)
            elapsed += ready - start
            if i == 0:
                startup.append(ready - begun)
            starts += batch.batch_size
            for key in counters:
                counters[key] += int(batch.metadata[key])
            edges = _edge_counts(batch.inputs["graph"])
            # Each action in this fixture names a distinct Pin for its owner.
            frontier_starts += edges["action_pin"] if args.history else batch.batch_size
            returned += sum(edges.values())
            if args.positives is not None:
                graph = batch.inputs["positive_graph"]
                if isinstance(graph, api.ColumnarGraph):
                    count = graph.candidate_offsets[-1].as_py()
                else:
                    count = graph["candidate_offsets"][-1].item()
                if count != batch.batch_size * args.positives:
                    raise AssertionError("unexpected candidate count")
                positive_count += count
                positive_edges += sum(_edge_counts(graph).values())
                profile = source.last_graph_profile
                if profile["samples"] != count:
                    raise AssertionError("graph timing does not match candidate roots")
                sample_times.append(profile["sampling_ns"] / 1e6)
                graph_times.append(profile["total_ns"] / 1e6)
        start = perf_counter()
        if next(iterator, None) is not None:
            raise AssertionError("unexpected extra anchor batch")
        elapsed += perf_counter() - start
    ordered = sorted(latencies)
    return {
        "roots": starts,
        "frontier_starts": frontier_starts,
        **counters,
        **(
            {
                "candidate_roots": positive_count,
                "candidate_graphs_per_second": positive_count / elapsed,
                "candidate_edges": positive_edges,
            }
            if args.positives is not None
            else {}
        ),
        **(
            {
                "p50_candidate_sampling_ms": statistics.median(sample_times),
                "p50_candidate_graph_ms": statistics.median(graph_times),
            }
            if graph_times
            else {}
        ),
        "output_edges": returned,
        "seconds": elapsed,
        "roots_per_second": starts / elapsed,
        "time_to_first_batch_ms": statistics.median(startup) * 1000,
        "p50_batch_ms": 1000 * statistics.median(latencies),
        "p95_batch_ms": 1000 * ordered[math.ceil(len(ordered) * 0.95) - 1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("diskgraph",), default="diskgraph")
    parser.add_argument(
        "--positives",
        type=int,
        help="caller-supplied Pins per anchor; omit for an anchor-only baseline",
    )
    parser.add_argument(
        "--output-format", choices=("model", "columnar", "inspectable"), default="model"
    )
    parser.add_argument("--mode", choices=("walk", "hop"), default="walk")
    parser.add_argument(
        "--history", action="store_true", help="walk from up to four user history Pins"
    )
    for name, default in (
        ("pins", 1024),
        ("batch-size", 16),
        ("batches", 8),
        ("repeats", 3),
        ("budget", 1024),
        ("threads", 1),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument(
        "--json", dest="json_path", type=Path, help="also write the result to this file"
    )
    args = parser.parse_args()
    if args.positives is not None and not 0 <= args.positives <= args.pins:
        parser.error("positives must be between zero and the number of Pins")
    if any(
        value <= 0
        for name, value in vars(args).items()
        if name != "positives" and type(value) is int
    ):
        parser.error("sizes, repeats, thread count and walk budget must be positive")
    output = json.dumps(measure(args), sort_keys=True)
    print(output)
    if args.json_path:
        args.json_path.write_text(output + "\n")


if __name__ == "__main__":
    main()
