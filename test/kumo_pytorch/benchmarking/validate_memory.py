"""Validate real Linux stage peaks, including transient child allocations.

Run only inside an already provisioned dedicated writable cgroup. This command
never provisions groups, changes limits, migrates processes or remounts anything.
"""

import argparse
import hashlib
import io
import json
import mmap
import platform
import subprocess
import sys
import traceback
from pathlib import Path

from .memory import CgroupMemory
from .recording import Recorder


def allocate(size):
    with mmap.mmap(-1, size) as memory:
        for offset in range(0, size, 4096):
            memory[offset] = 1


def validate(cgroup):
    cgroup = Path(cgroup)
    collector = CgroupMemory(cgroup)
    collector.validate()
    stream = io.StringIO()
    recorder = Recorder(stream, observers=(collector,))
    baseline = int((cgroup / "memory.current").read_text())
    with recorder.span("ingestion"):
        allocate(64 * 1024**2)
    with recorder.span("materialization"):
        allocate(4 * 1024**2)
    child_baseline = int((cgroup / "memory.current").read_text())
    with recorder.span("batch.request"):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "memory = bytearray(48 * 1024**2); print(len(memory))",
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
        raise AssertionError(f"Kernel peaks unavailable: {peaks}")
    high = peaks["ingestion.peak_host_bytes"]["value"]
    low = peaks["materialization.peak_host_bytes"]["value"]
    child = peaks["batch_production.peak_host_bytes"]["value"]
    if high < baseline + 48 * 1024**2:
        raise AssertionError("Transient allocation not retained by the stage peak")
    if low >= high - 32 * 1024**2:
        raise AssertionError("New stage inherited the previous stage's high-water mark")
    if child < child_baseline + 32 * 1024**2:
        raise AssertionError("Exited child's transient allocation was not counted")
    return {
        "baseline_bytes": baseline,
        "child_baseline_bytes": child_baseline,
        "observations": records,
        "checks": [
            "transient_parent_allocation",
            "independent_stage_reset",
            "transient_child_allocation",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgroup", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
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
        result.update(validate(args.cgroup), status="passed")
    except Exception:  # noqa: BLE001 -- persist real validation failures; never skip
        result.update(status="failed", traceback=traceback.format_exc())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
