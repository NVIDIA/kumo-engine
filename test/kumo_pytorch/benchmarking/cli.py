# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI options and result presentation for benchmark execution layers."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def worker_command(**options):
    return [
        sys.executable,
        "-m",
        "test.kumo_pytorch.benchmark",
        *[
            part
            for key, value in options.items()
            for part in ("--" + key.replace("_", "-"), str(value))
        ],
    ]


def main(execute, argv, *, scope, description):
    parser = argparse.ArgumentParser(description=description)
    if scope != "suite":
        parser.add_argument(
            "--adapter", required=True, help="Trusted Python module:factory"
        )
    parser.add_argument("--config", required=True, type=Path, help="Configuration JSON")
    parser.add_argument("--output", required=True, type=Path, help="New result directory")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--gpu-interval",
        type=float,
        default=1.0 if scope == "worker" else 0.1,
        help="GPU utilization sampling interval in seconds",
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
    if scope == "worker":
        parser.add_argument("--warmup-batches", type=int, default=2)
        parser.add_argument("--steady-min-batches", type=int, default=4)
        parser.add_argument(
            "--memory-cgroup",
            type=Path,
            default=os.environ.get("KUMO_BENCHMARK_CGROUP"),
            help="Existing Linux cgroup v2 containing only this workload and children",
        )
    else:
        parser.add_argument("--index-lifecycle", action="store_true")
        parser.add_argument("--worker-timeout", dest="timeout", type=float, default=3600)
        parser.add_argument("--isolate-memory", action="store_true")
        if scope == "campaign":
            parser.add_argument("--compare-fitting", action="store_true")
    options = vars(parser.parse_args(argv))
    adapter = options.pop("adapter", None)
    if scope == "worker":
        module, separator, name = adapter.partition(":")
        if not separator or not module or not name:
            parser.error("--adapter must be module:factory")
        adapter = getattr(importlib.import_module(module), name)
    output = options.pop("output")
    inputs = [json.loads(options.pop("config").read_text()), output]
    if scope == "suite":
        options["compare_fitting"] = not options["index_lifecycle"]
    else:
        inputs.insert(0, adapter)
    result = execute(*inputs, **options)
    if scope == "worker":
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
        print(f"Report: {(output / 'report.md').resolve()}")
    else:
        print(f"{result['status']}: {output / 'report.md'}")
    return {"complete": 0, "failed": 1, "incomplete": 2}[result["status"]]
