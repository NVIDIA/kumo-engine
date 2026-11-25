# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time local and loopback S3 Parquet through the same public DiskGraph APIs."""

import argparse
import json
import math
import platform
import statistics
import tempfile
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import diskgraph
import kumo.settings
import pyarrow as pa
import pyarrow.parquet as pq

from kumo.pytorch import api
from tests.e2e.s3_endpoint import local_s3
from tests.test_scored_walks import star


def inputs(root, client, pins, anchors):
    schema, _ = star()
    events = list(range(pins * 8))
    tables = {
        "pins": pa.table({"id": range(pins)}),
        "boards": pa.table({"id": range(128)}),
        "membership": pa.table(
            {
                "id": events,
                "pin": [i // 8 for i in events],
                "board": [i % 128 for i in events],
                "at": pa.array([1 + i % 8 for i in events], type=pa.timestamp("s", tz="UTC")),
            }
        ),
        "anchors": pa.table(
            {
                "sid": range(anchors),
                "entity": [(i * 131) % pins for i in range(anchors)],
                "at": pa.array(
                    [4 if i % 2 else 8 for i in range(anchors)], type=pa.timestamp("s", tz="UTC")
                ),
            }
        ),
    }
    sources = {"local": {}, "s3": {}}
    files = {}
    for name, table in tables.items():
        folder = root / name
        folder.mkdir()
        objects = []
        num_parts = 4 if name == "membership" else 2 if name == "anchors" else 1
        for i in range(num_parts):
            start, stop = len(table) * i // num_parts, len(table) * (i + 1) // num_parts
            path = folder / f"part-{i}.parquet"
            pq.write_table(table.slice(start, stop - start), path)
            key = f"{name}/{path.name}"
            client.put_object(Bucket="omnisage-fixture", Key=key, Body=path.read_bytes())
            objects.append(path.stat().st_size)
        files[name] = {"rows": len(table), "parts": len(objects), "bytes": sum(objects)}
        sources["local"][name] = api.Parquet(str(folder))
        sources["s3"][name] = api.Parquet(f"s3://omnisage-fixture/{name}/")
    configs = {}
    for label, source in sources.items():
        anchor_table = api.AnchorTable(source.pop("anchors"), "entity", "at", sample_id="sid")
        configs[label] = api.Dataset(source), anchor_table
    return schema, configs, files


def verify(task, pins, expected):
    count = 0
    for batch in task.batches(output_format="columnar"):
        graph = batch.inputs["graph"]
        sids = batch.metadata["sample_ids"]
        assert list(sids) == list(range(count, count + len(sids)))
        root = graph.nodes["pins"]
        assert root.ids.to_pylist() == [(sid * 131) % pins for sid in sids]
        edges = graph.edges["pin_links"].index
        members = graph.nodes["membership"].ids.to_pylist()
        actual = [[] for _ in sids]
        for src, dst in zip(*(col.to_pylist() for col in edges.columns), strict=True):
            actual[src].append(members[dst])
        for sid, member_ids in zip(sids, actual, strict=True):
            base = ((sid * 131) % pins) * 8
            assert sorted(member_ids) == list(range(base, base + (4 if sid % 2 else 8)))
        count += len(sids)
    assert count == expected


def run_case(spec, config, samples, args):
    dataset, anchors = config
    engine = api.DiskGraphEngine(num_threads=1)
    start = perf_counter_ns()
    with api.prepare(spec, dataset, engine):
        samples["cold_prepare_ms"].append((perf_counter_ns() - start) / 1e6)
    with patch.object(diskgraph, "ingest", side_effect=AssertionError("warm index rebuilt")):
        start = perf_counter_ns()
        plan = api.prepare(spec, dataset, engine)
        samples["reuse_prepare_ms"].append((perf_counter_ns() - start) / 1e6)
    with plan:
        start = perf_counter_ns()
        with plan.materialize(anchors) as task:
            samples["materialize_anchors_ms"].append((perf_counter_ns() - start) / 1e6)
            verify(task, args.pins, args.anchors)  # Also warms the local index for epoch timing.
            start = previous = perf_counter_ns()
            count = 0
            for batch in task.batches(output_format="columnar"):
                now = perf_counter_ns()
                samples["batch_ms"].append((now - previous) / 1e6)
                previous = now
                count += len(batch.metadata["sample_ids"])
            elapsed = (perf_counter_ns() - start) / 1e9
            assert count == args.anchors
            samples["epoch_ms"].append(elapsed * 1000)
            samples["anchors_per_second"].append(count / elapsed)


def measure(args):
    metrics = {label: defaultdict(list) for label in ("local", "s3")}
    with local_s3() as client, tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        kumo.settings.configure(scratch_dir=root / "scratch")
        schema, configs, files = inputs(root, client, args.pins, args.anchors)
        hop = api.GraphLayer((api.GraphSample("pin_links", 8, strategy="recent"),))
        graph = api.NeighborhoodSpec((hop,))
        spec = api.DataLoaderSpec(
            schema,
            api.SampleSpec(api.EntitySpec("pins"), neighborhoods={"graph": graph}),
            api.BatchSpec(
                inputs={"graph": api.Input(api.NeighborhoodSource("graph"), api.Graph(), None)},
                batching=api.Batching(max_samples=args.batch_size),
            ),
        )
        for i in range(args.repeats):
            for label in configs if i % 2 == 0 else reversed(configs):
                run_case(spec, configs[label], metrics[label], args)
    return {
        "method": (
            "Matched Parquet bytes on local disk and loopback Moto HTTP. "
            "Each repetition builds a fresh local index. Warm prepare uses the same engine "
            "with dataset ingest disabled. Every sampled membership is checked against its "
            "anchor cutoff before timing a warm columnar epoch. Setup and upload excluded."
        ),
        "system": {
            "os": platform.platform(),
            "python": platform.python_version(),
            **{name: version(name) for name in ("boto3", "moto", "pyarrow")},
        },
        "config": {**vars(args), "diskgraph_threads": 1, "shards": 1, "tables": files},
        "metrics": {
            label: {
                metric: {
                    "n": len(values),
                    "p50": statistics.median(values),
                    "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
                    **({"runs": values} if metric != "batch_ms" else {}),
                }
                for metric, values in results.items()
            }
            for label, results in metrics.items()
        },
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pins", type=int, default=32768)
    cli.add_argument("--anchors", type=int, default=8192)
    cli.add_argument("--batch-size", type=int, default=64)
    cli.add_argument("--repeats", type=int, default=5)
    args = cli.parse_args()
    if min(args.pins, args.batch_size, args.repeats) < 1 or args.anchors < 2:
        cli.error("sizes must be positive with at least two anchors")
    print(json.dumps(measure(args), indent=2))
