# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an engine-owned benchmark adapter and write readable/structured reports."""

import argparse
import importlib
import json
import os
from pathlib import Path

from .benchmarking import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True, help="Trusted Python module:factory"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Adapter configuration JSON"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New result directory"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--steady-min-batches", type=int, default=4)
    parser.add_argument(
        "--gpu-interval",
        type=float,
        default=1.0,
        help="GPU utilization sampling interval in seconds",
    )
    parser.add_argument(
        "--memory-cgroup",
        type=Path,
        default=os.environ.get("KUMO_BENCHMARK_CGROUP"),
        help="Existing Linux cgroup v2 containing only this workload and children",
    )
    parser.add_argument(
        "--memory-method",
        choices=("kernel", "sampled"),
        default="kernel",
        help="Kernel high-water peaks or explicitly labelled sampled cgroup maxima",
    )
    parser.add_argument(
        "--memory-interval",
        type=float,
        default=0.01,
        help="Seconds between sampled host-memory reads",
    )
    parser.add_argument(
        "--measurement-profile", choices=("full", "timing"), default="full"
    )
    args = parser.parse_args(argv)
    module, separator, name = args.adapter.partition(":")
    if not separator or not module or not name:
        parser.error("--adapter must be module:factory")
    factory = getattr(importlib.import_module(module), name)
    result = run(
        factory,
        json.loads(args.config.read_text()),
        args.output,
        repetitions=args.repetitions,
        epochs=args.epochs,
        memory_cgroup=args.memory_cgroup,
        gpu_interval=args.gpu_interval,
        memory_method=args.memory_method,
        memory_interval=args.memory_interval,
        measurement_profile=args.measurement_profile,
        warmup_batches=args.warmup_batches,
        steady_min_batches=args.steady_min_batches,
    )
    print(
        f"{result['status']}: {result['metric_coverage']['complete']}/"
        f"{result['metric_coverage']['total']} metric obligations covered"
    )
    for row in result["epochs"]:
        values = row["metrics"]
        print(
            f"repetition={row['scope']['repetition']} epoch={row['scope']['epoch']} "
            f"samples/s={values.get('samples_per_second', 'unavailable')} "
            f"p95_ns={values.get('batch_latency_p95_ns', 'unavailable')}"
        )
    print(f"Report: {(args.output / 'report.md').resolve()}")
    return {"complete": 0, "failed": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
