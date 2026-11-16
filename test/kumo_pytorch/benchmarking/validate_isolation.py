"""Exercise the real worker supervisor and benchmark collector on Linux."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .cgroup_run import execute
from .interface import BatchCounts, Workload
from .runner import run
from .validate_memory import allocate, validate


def factory(config, workdir):
    def prepare(recorder):
        with recorder.span("ingestion"):
            allocate(64 * 1024**2)
        with recorder.span("materialization"):
            allocate(4 * 1024**2)

    def batches(epoch):
        for _ in range(6):
            subprocess.run(
                [sys.executable, "-c", "data = bytearray(16 * 1024**2)"], check=True
            )
            yield 1

    return Workload(
        identity={"dataset": "memory-validation", "workload": "transient-allocation"},
        prepare=prepare,
        batches=batches,
        count=lambda value: BatchCounts(value, 0),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("/sys/fs/cgroup"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        group = Path(os.environ["KUMO_BENCHMARK_CGROUP"])
        validation = validate(group)
        result = run(
            factory,
            {},
            args.output / "worker",
            repetitions=1,
            epochs=2,
            memory_cgroup=group,
        )
        assert result["execution_status"] == "completed", result["errors"]
        for name in ("ingestion", "materialization", "batch_production"):
            rows = result["measurements"][name + ".peak_host_bytes"]
            assert rows and all(row["status"] == "available" for row in rows)
        validation.update(
            uid=os.getuid(),
            cgroup=str(group),
            checks=validation["checks"] + ["complete_worker_stages", "two_epochs"],
        )
        (args.output / "worker-validation.json").write_text(
            json.dumps(validation, indent=2)
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.chmod(0o777)  # disposable validation output, writable by test UID
    previous = set(args.parent.glob("kumo-benchmark-*"))
    code = execute(
        [sys.executable, "-m", __spec__.name, "--worker", "--output", str(args.output)],
        parent=args.parent,
        uid=65534,
        gid=65534,
        timeout=60,
    )
    assert code == 0, f"Worker exited {code}"
    worker = json.loads((args.output / "worker-validation.json").read_text())
    assert worker["uid"] == 65534
    assert (
        execute(
            [
                sys.executable,
                "-c",
                "import subprocess,time; subprocess.Popen(['sleep','30']); time.sleep(30)",
            ],
            parent=args.parent,
            uid=65534,
            gid=65534,
            timeout=0.2,
        )
        == 124
    )
    assert set(args.parent.glob("kumo-benchmark-*")) == previous
    result = {
        "status": "passed",
        "kernel": platform.release(),
        "checks": worker["checks"]
        + ["unprivileged_worker", "timeout_kills_descendants", "cgroup_cleanup"],
    }
    (args.output / "isolation-validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
