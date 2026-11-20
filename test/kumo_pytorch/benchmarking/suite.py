# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a declared collection of dataset/model-independent benchmark campaigns."""

import re
import traceback
from pathlib import Path

from .campaign import run_campaign
from .cli import main as run_cli
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
    if options.get("measurement_profile") == "timing":
        from .timing_report import timing_summary

        lines += ["", *timing_summary(cases, output)]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    return run_cli(run_suite, argv, scope="suite", description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
