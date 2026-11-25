# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure delivery of a dense float32 node feature through native Parquet graph batches.

Compare --storage array and --storage scalar for identical data, projections and
temporal sampling. The scalar case includes the stack needed to make the same
model input. Index construction and model execution are outside the batch timer.
"""

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import kumo.settings
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from kumo.pytorch import api


def fixture(path, width, storage, batch_size, batches, fanout):
    boards = 128
    pins = np.arange(boards * 16, dtype=np.int64)
    # Exact float32 integers make both the per-node oracle and the scalar
    # comparison independent of floating-point reduction order.
    matrix = (pins[:, None] % 257 + np.arange(width) % 257).astype(np.float32)
    columns = {
        "id": pa.array(pins),
        "board": pa.array(pins // 16),
        "at": pa.array(pins % 16 + 1, type=pa.timestamp("s", tz="UTC")),
    }
    if storage == "array":
        fields = {"embedding": api.DType("float32", shape=(width,))}
        columns["embedding"] = pa.ListArray.from_arrays(
            pa.array(np.arange(len(pins) + 1, dtype=np.int32) * width), pa.array(matrix.ravel())
        )
    else:
        fields = {f"v_{i}": api.DType("float32") for i in range(width)}
        columns.update((name, pa.array(matrix[:, i])) for i, name in enumerate(fields))
    pq.write_table(pa.table(columns), path / "pins.parquet")
    pq.write_table(pa.table({"id": np.arange(boards)}), path / "boards.parquet")
    rows = np.arange(batch_size * batches)
    pq.write_table(
        pa.table(
            {
                "board": rows * 13 % boards,
                "at": pa.array(np.where(rows % 2 == 0, 12, 16), type=pa.timestamp("s", tz="UTC")),
            }
        ),
        path / "anchors.parquet",
    )
    integer = api.DType("int64")
    schema = api.DataSchema(
        (
            api.TableSpec("boards", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "pins",
                "id",
                {
                    "id": integer,
                    "board": integer,
                    "at": api.DType("timestamp", unit="s", timezone="UTC"),
                    **fields,
                },
                time_column="at",
            ),
        ),
        (api.RelationSpec("pins_of", "boards", "id", "pins", "board"),),
    )
    graph = api.Graph({"pins": api.Fields(tuple(api.Encode(f, api.Raw()) for f in fields))})
    spec = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec("boards"),
            neighborhoods={
                "pins": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("pins_of", fanout),)),)
                )
            },
        ),
        api.BatchSpec(
            inputs={"pins": api.Input(api.NeighborhoodSource("pins"), graph, None)},
            batching=api.Batching(max_samples=batch_size),
        ),
    )
    dataset = api.Dataset({t: api.Parquet(str(path / f"{t}.parquet")) for t in ("boards", "pins")})
    anchors = api.AnchorTable(api.Parquet(str(path / "anchors.parquet")), "board", "at")
    return spec, dataset, anchors, tuple(fields)


def measure(args):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        kumo.settings.configure(scratch_dir=path / "scratch")
        spec, dataset, anchors, fields = fixture(
            path, args.width, args.storage, args.batch_size, args.batches, args.fanout
        )
        started = perf_counter()
        with (
            api.prepare(spec, dataset, api.DiskGraphEngine(num_threads=args.threads)) as plan,
            plan.materialize(anchors) as task,
        ):
            setup = perf_counter() - started
            # Both physical layouts must use the native anchor iterator.
            with (
                patch.object(
                    task._anchors, "_requests", side_effect=AssertionError("Python anchor fallback")
                ),
                patch.object(
                    task._assembler, "batches", side_effect=AssertionError("Python batch fallback")
                ),
            ):

                def run(verify):
                    latencies, sampled, seconds = [], 0, 0.0
                    start = perf_counter()
                    iterator = iter(task.batches(output_format="model", shuffle=False))
                    seconds += perf_counter() - start
                    for _ in range(args.batches):
                        start = perf_counter()
                        graph = next(iterator).inputs["pins"]
                        nodes = graph["nodes"]["pins"]
                        features = nodes["features"]
                        x = (
                            features["embedding"]["values"]
                            if args.storage == "array"
                            else torch.stack([features[f]["values"] for f in fields], dim=-1)
                        )
                        elapsed = perf_counter() - start
                        latencies.append(elapsed)
                        seconds += elapsed
                        sampled += len(x)
                        if verify:
                            ids = nodes["ids"]
                            expected = (ids[:, None] % 257 + torch.arange(args.width) % 257).float()
                            assert x.shape == (args.batch_size * args.fanout, args.width)
                            assert torch.equal(x, expected)
                            assert all(feature["valid"].all() for feature in features.values())
                    start = perf_counter()
                    assert next(iterator, None) is None
                    seconds += perf_counter() - start
                    return latencies, sampled, seconds

                first, _, cold = run(True)
                times, total, duration = [], 0, 0.0
                for _ in range(args.repeats):
                    latency, count, elapsed = run(False)
                    times.extend(latency)
                    total += count
                    duration += elapsed
    return {k: v for k, v in vars(args).items() if k != "json"} | {
        "setup_seconds": setup,
        "first_batch_ms": 1000 * first[0],
        "first_pass_loader_seconds": cold,
        "warm_batches": len(times),
        "rows": total,
        "loader_seconds": duration,
        "p50_batch_ms": 1000 * statistics.median(times),
        "p95_batch_ms": 1000 * sorted(times)[math.ceil(0.95 * len(times)) - 1],
        "vectors_per_second": total / duration,
        "feature_mib_per_second": total * args.width * 4 / duration / 2**20,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--storage", choices=("array", "scalar"), required=True)
    for key, default in (
        ("width", 128),
        ("batch-size", 16),
        ("batches", 12),
        ("fanout", 8),
        ("repeats", 4),
        ("threads", 1),
    ):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    if any(v < 1 for v in vars(args).values() if isinstance(v, int)) or args.fanout > 12:
        p.error("sizes must be positive and fanout must be at most 12 (the earliest cutoff)")
    torch.set_num_threads(1)
    result = json.dumps(measure(args), sort_keys=True)
    if args.json:
        args.json.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
