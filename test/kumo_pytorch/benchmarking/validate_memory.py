# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate real Linux stage peaks, including transient child allocations.

Run only inside an already provisioned dedicated writable cgroup. This command
never provisions groups, changes limits, migrates processes or remounts anything.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .memory import CgroupMemory, SampledAnonymousMemory, SampledCgroupMemory
from .recording import Recorder


def allocate(size, hold=0):
    memory = bytearray(size)
    for offset in range(0, size, 4096):
        memory[offset] = 1
    if hold:
        time.sleep(hold)


def validate(cgroup, method="kernel"):
    cgroup = Path(cgroup)
    collector = (
        SampledCgroupMemory(cgroup) if method == "sampled" else CgroupMemory(cgroup)
    )
    hold = 0.2  # Anonymous peaks are sampled even with kernel total-memory peaks.
    collector.validate()
    stream = io.StringIO()
    recorder = Recorder(stream, observers=(collector, SampledAnonymousMemory(cgroup)))
    baseline = int((cgroup / "memory.current").read_text())
    with recorder.span("ingestion"):
        allocate(64 * 1024**2, hold)
    with recorder.span("materialization"):
        allocate(4 * 1024**2, hold)
    child_baseline = int((cgroup / "memory.current").read_text())
    with recorder.span("batch.request"):
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"import time; memory = bytearray(48 * 1024**2); time.sleep({hold}); print(len(memory))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    peaks = {
        row["metric"]: row
        for row in records
        if row["metric"].endswith(".peak_host_bytes")
    }
    if not all(row["status"] == "available" for row in peaks.values()):
        raise AssertionError(f"{method} peaks unavailable: {peaks}")
    high = peaks["ingestion.peak_host_bytes"]["value"]
    low = peaks["materialization.peak_host_bytes"]["value"]
    child = peaks["batch_production.peak_host_bytes"]["value"]
    if high < baseline + 48 * 1024**2:
        raise AssertionError("Transient allocation not retained by the stage peak")
    if low >= high - 32 * 1024**2:
        raise AssertionError("New stage inherited the previous stage's high-water mark")
    if child < child_baseline + 32 * 1024**2:
        raise AssertionError("Exited child's transient allocation was not counted")
    anon = {
        r["metric"]: r for r in records if r["metric"].endswith(".peak_anonymous_bytes")
    }
    if len(anon) != 3 or any(r["status"] != "available" for r in anon.values()):
        raise AssertionError(f"Anonymous memory unavailable: {anon}")
    high_anon = anon["ingestion.peak_anonymous_bytes"]["value"]
    low_anon = anon["materialization.peak_anonymous_bytes"]["value"]
    child_anon = anon["batch_production.peak_anonymous_bytes"]["value"]
    if high_anon < low_anon + 32 * 1024**2 or child_anon < low_anon + 24 * 1024**2:
        raise AssertionError(
            "Anonymous collector missed parent/child allocations or retained a prior peak"
        )
    return {
        "anonymous_checks": "independent sampled parent and descendant allocation peaks",
        "method": method,
        "held_allocation_seconds": hold,
        "baseline_bytes": baseline,
        "child_baseline_bytes": child_baseline,
        "observations": records,
        "checks": [
            "held_parent_allocation" if hold else "transient_parent_allocation",
            "independent_stage_measurement" if hold else "independent_stage_reset",
            "held_child_allocation" if hold else "transient_child_allocation",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cgroup", type=Path, default=os.environ.get("KUMO_BENCHMARK_CGROUP")
    )
    parser.add_argument("--method", choices=("kernel", "sampled"), default="kernel")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.cgroup is None:
        parser.error("--cgroup or KUMO_BENCHMARK_CGROUP is required")
    if args.output.exists():
        parser.error("Output already exists")
    result = {
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python": sys.version,
        "cgroup": str(args.cgroup),
        "source_hashes": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("memory.py", "recording.py", "validate_memory.py")
        },
    }
    try:
        result.update(validate(args.cgroup, args.method), status="passed")
    except Exception:  # noqa: BLE001 -- persist real validation failures; never skip
        result.update(status="failed", traceback=traceback.format_exc())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
