# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local vs loopback-feeder overhead for one contract (prototype measurement, not a harness).

    python -m e2e.datasets.hm --scale M --seed 0 --out /data/hm-m
    python benchmarks/remote_feeder_overhead.py --data /data/hm-m --index /data/hm-m-index \
        --contract gnn --mode remote

Run each mode in its own process so peak RSS is that mode's alone.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from e2e.flows import hm_gnn, hm_pragma
from e2e.flows.common import anchors, dataset
from kumo.pytorch.api import DiskGraphEngine, prepare
from kumo.pytorch.engine import Engine
from kumo.pytorch.engines.distributed import Cluster, DistributedEngine
from kumo.pytorch.engines.distributed.wire import encode_batch

CONTRACTS = {"gnn": hm_gnn.contract, "sequence": hm_pragma.pretraining_contract}


def _peak_rss_mb(pid: int | None = None) -> float:
    if pid is None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"no VmHWM for {pid}")


def measure(engine: Engine, args: argparse.Namespace) -> dict[str, float]:
    contract = CONTRACTS[args.contract]()
    with prepare(contract, dataset(args.data), engine) as plan:
        started = time.perf_counter()
        task = plan.materialize(anchors(args.data), fit_split="train")
        materialized = time.perf_counter()
        batches = task.batches(
            split="train", batch_size=args.batch_size, shuffle=True, output_format="model"
        )
        next(batches)
        first_at = time.perf_counter()
        for index, _ in enumerate(batches, start=1):
            if index == args.batches:
                break
        finished = time.perf_counter()
        sample = task.batches(split="train", batch_size=args.batch_size, output_format="model")
        sizes = [len(encode_batch(batch)) for _, batch in zip(range(10), sample)]
        return {
            "materialize_s": materialized - started,
            "first_batch_s": first_at - materialized,
            "batches_per_s": args.batches / (finished - first_at),
            "bytes_per_batch": sum(sizes) / len(sizes),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--contract", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    inner = DiskGraphEngine(index=args.index, num_threads=args.num_threads)
    if args.mode == "local":
        result = measure(inner, args)
        result["driver_peak_rss_mb"] = _peak_rss_mb()
    else:
        with DistributedEngine(inner, cluster=Cluster.subprocess(feeders=1)) as engine:
            result = measure(engine, args)
            result["driver_peak_rss_mb"] = _peak_rss_mb()
            result["feeder_peak_rss_mb"] = _peak_rss_mb(engine.feeder_pids[0])
    print(json.dumps({"contract": args.contract, "mode": args.mode, **result}))


if __name__ == "__main__":
    main()
