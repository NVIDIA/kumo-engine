"""Run a declared collection of dataset/model-independent benchmark campaigns."""

import argparse
import json
import re
import traceback
from pathlib import Path

from .campaign import run_campaign
from .reporting import write_json


def run_suite(cases, output, **options):
    names = [case["name"] for case in cases]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) for name in names)
    ):
        raise ValueError("Cases require unique, safe directory names")
    if any(set(case) != {"name", "adapter", "config"} for case in cases):
        raise ValueError("Each case requires name, adapter and config")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"cases": cases, "options": options})
    rows = []
    for case in cases:
        try:
            result = run_campaign(
                case["adapter"], case["config"], output / case["name"], **options
            )
            row = {
                "name": case["name"],
                "status": result["status"],
                "report": f"{case['name']}/report.md",
            }
        except Exception:  # noqa: BLE001 -- retain failed case without skipping later workloads
            row = {
                "name": case["name"],
                "status": "failed",
                "traceback": traceback.format_exc(),
                "report": None,
            }
        rows.append(row)
        # Retain completed cases even if a later run is interrupted.
        write_json(output / "results.json", {"status": "running", "cases": rows})
    status = (
        "failed"
        if any(row["status"] == "failed" for row in rows)
        else "complete"
        if all(row["status"] == "complete" for row in rows)
        else "incomplete"
    )
    result = {"status": status, "cases": rows}
    write_json(output / "results.json", result)
    lines = [
        "# Nightly benchmark suite",
        "",
        f"Status: **{status}**. Completeness is a measurement gate, not a performance threshold.",
        "",
        "| Case | Status | Report |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        link = (
            f"[campaign]({row['report']})"
            if row["report"]
            else "See results.json failure"
        )
        lines.append(f"| {row['name']} | {row['status']} | {link} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--worker-timeout", type=float, default=3600)
    parser.add_argument("--isolate-memory", action="store_true")
    parser.add_argument("--gpu-interval", type=float, default=0.1)
    args = parser.parse_args(argv)
    result = run_suite(
        json.loads(args.config.read_text()),
        args.output,
        repetitions=args.repetitions,
        epochs=args.epochs,
        timeout=args.worker_timeout,
        isolate_memory=args.isolate_memory,
        gpu_interval=args.gpu_interval,
        compare_fitting=True,
    )
    print(f"{result['status']}: {args.output / 'report.md'}")
    return {"complete": 0, "failed": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
