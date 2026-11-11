# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local vs feeder throughput for one contract (prototype measurement, not a harness).

    python -m e2e.datasets.hm --scale M --seed 0 --out /data/hm-m
    python benchmarks/remote_feeder_overhead.py --data /data/hm-m --index /data/hm-m-index \
        --contract gnn --mode remote --feeders 2 --stripes 4 --fitted /data/hm-m-gnn-state

Run each configuration in its own process so peak RSS is that configuration's alone.
``--fitted`` reuses one fit across runs: the first run saves it, later runs load it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from e2e.flows import hm_gnn, hm_pragma
from e2e.flows.common import anchors, dataset
from kumo.pytorch.api import Batch, DiskGraphEngine, prepare
from kumo.pytorch.artifact import Artifact, FittedState
from kumo.pytorch.engine import Engine
from kumo.pytorch.engines.distributed import Cluster, DistributedEngine
from kumo.pytorch.engines.distributed.wire import encode_batch
from kumo.pytorch.task import Task

CONTRACTS = {"gnn": hm_gnn.contract, "sequence": hm_pragma.pretraining_contract}
_TICKS = os.sysconf("SC_CLK_TCK")


def _peak_rss_mb(pid: int | None = None) -> float:
    if pid is None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"no VmHWM for {pid}")


def _cpu_seconds(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / _TICKS


def _stream(task: Task, batch_size: int) -> Iterator[Batch]:
    """A training loop's batches: shuffled epochs back to back."""
    for epoch in itertools.count():
        yield from task.batches(
            split="train", batch_size=batch_size, epoch=epoch, shuffle=True, output_format="model"
        )


def _fitted(path: Path | None) -> FittedState | None:
    return Artifact.load(path).fitted_state if path is not None and path.exists() else None


def measure(engine: Engine, args: argparse.Namespace, pids: list[int]) -> dict[str, float]:
    contract = CONTRACTS[args.contract]()
    fitted = _fitted(args.fitted)
    with prepare(contract, dataset(args.data), engine) as plan:
        started = time.perf_counter()
        if fitted is None:
            task = plan.materialize(anchors(args.data), fit_split="train")
            if args.fitted is not None:
                task.to_artifact().save(args.fitted)
        else:
            task = plan.materialize(anchors(args.data), fitted=fitted)
        materialized = time.perf_counter()
        batches = _stream(task, args.batch_size)
        next(batches)
        first_at = time.perf_counter()
        for _ in range(args.warmup):
            next(batches)
        cpu_before = {pid: _cpu_seconds(pid) for pid in pids}
        driver_before = time.process_time()
        window_start = time.perf_counter()
        for batch in itertools.islice(batches, args.batches):
            if args.device != "cpu":
                batch.to(args.device)
        if args.device != "cpu":
            torch.cuda.synchronize()
        window = time.perf_counter() - window_start
        feeder_cpu = sum(_cpu_seconds(pid) - cpu_before[pid] for pid in pids)
        driver_cpu = time.process_time() - driver_before
        batches.close()
        sample = task.batches(split="train", batch_size=args.batch_size, output_format="model")
        sizes = [len(encode_batch(batch)) for batch in itertools.islice(sample, 10)]
        return {
            "train_batches_per_epoch": -(-task.count("train") // args.batch_size),
            "materialize_s": materialized - started,
            "first_batch_s": first_at - materialized,
            "batches_per_s": args.batches / window,
            "bytes_per_batch": sum(sizes) / len(sizes),
            "feeder_cpu_cores": feeder_cpu / window,
            "driver_cpu_cores": driver_cpu / window,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--contract", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    parser.add_argument("--feeders", type=int, default=1)
    parser.add_argument("--stripes", type=int, default=1)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=0, help="batches skipped before timing")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="move each batch here, e.g. cuda")
    parser.add_argument("--fitted", type=Path, help="artifact directory to reuse a fit from")
    args = parser.parse_args()
    inner = DiskGraphEngine(index=args.index, num_threads=args.num_threads)
    config = {"contract": args.contract, "mode": args.mode, "device": args.device}
    if args.mode == "local":
        result = measure(inner, args, [])
        result["driver_peak_rss_mb"] = _peak_rss_mb()
    else:
        config |= {"feeders": args.feeders, "stripes": args.stripes}
        with DistributedEngine(
            inner, cluster=Cluster.subprocess(feeders=args.feeders), stripes=args.stripes
        ) as engine:
            engine.start()
            pids = [pid for feeder in engine.producer_pids for pid in feeder]
            result = measure(engine, args, pids)
            result["driver_peak_rss_mb"] = _peak_rss_mb()
            peaks = [_peak_rss_mb(pid) for pid in pids]
            result["producer_peak_rss_mb_max"] = max(peaks)
            result["producer_peak_rss_mb_sum"] = sum(peaks)
    print(json.dumps({**config, **result}))


if __name__ == "__main__":
    main()
