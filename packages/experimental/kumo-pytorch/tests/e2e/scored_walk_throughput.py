# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time DiskGraph candidate selection and graph delivery on a synthetic Pin dataset."""

import argparse
import json
import math
import resource
import statistics
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import kumo.settings
import pyarrow as pa
import pyarrow.parquet as pq

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
        if args.draw == "future":
            for user in range(root_count):
                for offset in range(1, 9):
                    rows["actions"].append(
                        dict(
                            id=len(rows["actions"]),
                            user=user,
                            pin=(4 * user + 4 + offset) % args.pins,
                            at=cutoff + offset,
                        )
                    )
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
        draw = None
        if args.draw == "neighbors":
            draw = api.DrawNeighbors("graph", ("ranked.pins",), args.positives, weight="score")
        elif args.draw == "future":
            draw = api.DrawRows(
                api.Rows(("user_actions",), api.TimeWindow("at", start=0, stop=8)),
                column="pin",
                count=args.positives,
            )
        positive = api.CandidateSetSpec(
            api.EntitySpec("pins"), neighborhoods={"graph": api.NeighborhoodSpec(layers)}, draw=draw
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
    if args.positives is not None and args.draw is None:
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
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        kumo.settings.configure(scratch_dir=root / "scratch")
        writer = (
            parquet_inputs if args.positives is not None and args.draw is None else write_parquet
        )
        dataset, anchors = writer(schema, rows, requests, root)
        engine = api.DiskGraphEngine(num_threads=args.threads)
        primary = stack.enter_context(_open_task(contract, dataset, engine, anchors))
        first_pass = _summary([_record_pass(primary, args)])
        tasks = {"primary": primary}
        supplied_first = groups = None
        if args.draw is not None:
            # Capture the exact ordered IDs separately from either timed run.
            groups = {}
            for batch in _batches(primary, args):
                groups.update(
                    zip(
                        batch.metadata["sample_ids"],
                        batch.metadata["candidate_entity_ids"]["positive"],
                        strict=True,
                    )
                )
            table = pq.read_table(anchors.source.path)
            ids = [[{"pin": pin} for pin in groups[sid]] for sid in table["sid"].to_pylist()]
            replay_path = root / "replay.parquet"
            column = pa.array(ids, type=pa.list_(pa.struct([("pin", pa.int64())])))
            pq.write_table(table.append_column("positives", column), replay_path)
            supplied_anchors = replace(
                anchors,
                source=api.Parquet(str(replay_path)),
                candidates={"positive": api.CandidateColumn("positives", "pin")},
            )
            supplied = replace(
                contract,
                sample=replace(
                    contract.sample,
                    candidate_sets={
                        "positive": replace(positive, draw=None),
                    },
                ),
            )
            replay = stack.enter_context(_open_task(supplied, dataset, engine, supplied_anchors))
            tasks["supplied"] = replay
            supplied_first = _summary([_record_pass(replay, args, expected=groups)])
            _check_work(first_pass, supplied_first)
        samples = {name: [] for name in tasks}
        for repeat in range(args.repeats):
            # Alternate the order to distribute thermal and cache drift.
            for name in reversed(tasks) if repeat % 2 else tasks:
                samples[name].append(_record_pass(tasks[name], args, groups))
        warm = _summary(samples["primary"])
        if supplied_first:
            comparison = _summary(samples["supplied"])
            _check_work(warm, comparison)
            warm["supplied_replay"] = {"first_pass": supplied_first, **comparison}
    return {
        **{key: value for key, value in vars(args).items() if key != "json_path"},
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "first_pass": first_pass,
        **warm,
    }


@contextmanager
def _open_task(contract, dataset, engine, anchors):
    refusal = RuntimeError("benchmark selected the Python fallback")
    with (
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(anchors) as task,
        patch.object(task._anchors, "_requests", side_effect=refusal),
        patch.object(task._assembler, "batches", side_effect=refusal),
    ):
        yield task


def _batches(task, args):
    return task.batches(output_format=args.output_format, shuffle=False, seed=17)


def _check_work(drawn, supplied):
    for key in (
        "roots",
        "frontier_starts",
        "output_edges",
        "candidate_roots",
        "candidate_edges",
        "walk_pushes",
        "walk_decoded_entries",
    ):
        if drawn.get(key) != supplied.get(key):
            raise AssertionError(
                f"drawn vs supplied workload differs for {key}: {drawn[key]} vs {supplied[key]}"
            )


def _edge_counts(graph):
    if isinstance(graph, api.ColumnarGraph):
        return {name: edge.index.num_rows for name, edge in graph.edges.items()}
    return {name: edge["index"].shape[1] for name, edge in graph["edges"].items()}


def _record_pass(task, args, expected=None):
    source = task._assembler._source
    ordered_ids = iter(expected) if expected is not None else None
    draw_times = []
    latencies, startup = [], []
    returned, starts, frontier_starts, elapsed = 0, 0, 0, 0.0
    positive_count = positive_edges = 0
    sample_times, graph_times = [], []
    counters = {"walk_pushes": 0, "walk_decoded_entries": 0} if args.mode == "walk" else {}
    begun = perf_counter()
    iterator = iter(_batches(task, args))
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
            groups = batch.metadata["candidate_entity_ids"]["positive"]
            if args.draw:
                if any(
                    len(group) > args.positives or len(set(group)) != len(group) for group in groups
                ):
                    raise AssertionError("draw exceeded its count or repeated a candidate")
                if count != sum(map(len, groups)):
                    raise AssertionError("candidate graphs do not match the selected IDs")
            elif count != batch.batch_size * args.positives:
                raise AssertionError("unexpected supplied candidate count")
            if ordered_ids is not None:
                ids = batch.metadata["sample_ids"]
                if tuple(ids) != tuple(next(ordered_ids) for _ in ids):
                    raise AssertionError("replay anchor order differs")
                if tuple(groups) != tuple(expected[sid] for sid in ids):
                    raise AssertionError("replay candidate IDs differ")
            positive_count += count
            positive_edges += sum(_edge_counts(graph).values())
            profile = source.last_graph_profile
            if profile["samples"] != count:
                raise AssertionError("graph timing does not match candidate roots")
            sample_times.append(profile["sampling_ns"] / 1e6)
            graph_times.append(profile["total_ns"] / 1e6)
        if draw_profile := source.last_candidate_draw_profile:
            if draw_profile["samples"] != batch.batch_size:
                raise AssertionError("draw timing does not match query roots")
            draw_times.append(draw_profile["total_ns"] / 1e6)
    start = perf_counter()
    if next(iterator, None) is not None:
        raise AssertionError("unexpected extra anchor batch")
    elapsed += perf_counter() - start

    work = {
        "roots": starts,
        "frontier_starts": frontier_starts,
        "output_edges": returned,
        **counters,
    }
    if args.positives is not None:
        work.update(candidate_roots=positive_count, candidate_edges=positive_edges)
    return work, elapsed, latencies, startup, sample_times, graph_times, draw_times


def _summary(passes):
    work, elapsed, batches, startup, sampling, graph, draws = zip(*passes, strict=True)
    counters = {key: sum(row[key] for row in work) for key in work[0]}
    elapsed = sum(elapsed)
    batch_times = sorted(t for group in batches for t in group)
    result = {
        **counters,
        "seconds": elapsed,
        "roots_per_second": counters["roots"] / elapsed,
        "time_to_first_batch_ms": statistics.median(t for group in startup for t in group) * 1000,
        "p50_batch_ms": statistics.median(batch_times) * 1000,
        "p95_batch_ms": batch_times[math.ceil(len(batch_times) * 0.95) - 1] * 1000,
    }
    if "candidate_roots" in counters:
        result.update(
            candidate_graphs_per_second=counters["candidate_roots"] / elapsed,
            p50_candidate_sampling_ms=statistics.median(t for group in sampling for t in group),
            p50_candidate_graph_ms=statistics.median(t for group in graph for t in group),
        )
    if any(draws):
        result["candidate_draw_p50_ms"] = statistics.median(t for group in draws for t in group)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("diskgraph",), default="diskgraph")
    parser.add_argument(
        "--draw",
        choices=("neighbors", "future"),
        help="compare engine-drawn Pins with replaying the identical IDs from Parquet",
    )
    parser.add_argument(
        "--positives",
        type=int,
        help="Pins per anchor (default 1 with --draw); omit both options for query graphs alone",
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
    if args.draw is not None:
        if args.positives is None:
            args.positives = 1
        if args.positives <= 0:
            parser.error("draws require a positive candidate count")
        if args.draw == "neighbors" and args.mode != "walk":
            parser.error("neighbor draws require --mode walk for ranked.pins")
        if args.draw == "future":
            args.history = True
    if args.draw is None and args.positives is not None and not 0 <= args.positives <= args.pins:
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
