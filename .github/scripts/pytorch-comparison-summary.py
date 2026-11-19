# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate comparison jobs without treating setup failures as package defects."""

import json
import os
import sys
from pathlib import Path

LABELS = {
    "original": "Original upstream suite",
    "package": "Expanded package checks",
    "fixtures": "Expanded dataset/model fixtures",
    "difficult": "Expanded Difficult Behavior",
}

REQUIRED = {
    "original": (
        "Original / Golden cases",
        "Original / Package tests",
        "Original / Style",
        "Original / Package build",
    ),
    "package": ("Expanded / Package checks",),
    "fixtures": ("Expanded / Dataset and model fixtures",),
    "difficult": ("Expanded / Difficult Behavior",),
}
TEST_STAGES = {title for titles in REQUIRED.values() for title in titles} - {
    "Original / Style",
    "Original / Package build",
}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def summarize(root):
    lines = [
        "# Original vs expanded test comparison",
        "",
        "All groups use the same package and DiskGraph runtime from main `44030297`. "
        "Expanded test assets are pinned to PR #353 at `5b1c9f2c`.",
        "",
        "Original tests run unchanged, including InMemory. Counts are check executions; "
        "upstream golden cases also occur in its package test run. "
        "Failures can expose package defects or test compatibility issues; "
        "the totals alone do not establish root cause.",
        "",
        "| Group / Stage | Job | Passed | Failed | Refused | Skipped |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    details = []
    failed = False
    for lane, label in LABELS.items():
        directory = root / f"comparison-results-{lane}"
        status_file = directory / "group-status.json"
        status = (
            json.loads(status_file.read_text())["status"]
            if status_file.exists()
            else "missing"
        )
        failed |= status != "success"
        steps = [
            json.loads(p.read_text()) for p in sorted(directory.rglob("step.json"))
        ]
        missing = set(REQUIRED[lane]) - {step["title"] for step in steps}
        failed |= bool(missing)
        for title in sorted(missing):
            lines.append(f"| {cell(title)} | missing or not executed | — | — | — | — |")
        for step in steps:
            failed |= step.get("status") != "passed" or bool(step.get("exit_code"))
            if step["title"] in TEST_STAGES and not step.get("tests"):
                failed = True
                lines.append(
                    f"| {cell(step['title'])} | {status}; no test report | — | — | — | — |"
                )
            counts = step.get("tests", {})
            errors = (
                counts.get("failed", 0)
                + counts.get("error", 0)
                + counts.get("errors", 0)
            )
            failed |= bool(errors)
            lines.append(
                f"| {cell(step['title'])} | {status} | {counts.get('passed', 0)} | "
                f"{errors} | {counts.get('refused (accepted)', 0)} | {counts.get('skipped', 0)} |"
            )
            cases = [
                c
                for c in step.get("checks", [])
                if c.get("status") in {"failed", "error"}
            ]
            if cases:
                details += [f"## {cell(step['title'])}", ""]
                for case in cases[:8]:
                    details.append(
                        f"- **{cell(case.get('title', case.get('check', 'Failure')))}**: {cell(case.get('reason', 'See the detailed artifact.'))}"
                    )
                if len(cases) > 8:
                    details.append(
                        f"- {len(cases) - 8} more failures are retained in this group's diagnostics."
                    )
                details.append("")
    lines += [
        "",
        *details,
        "Full logs, JUnit results and revision evidence are in each "
        "`comparison-results-*` artifact; readable detailed reports are in "
        "`comparison-diagnostics-*`.",
        "",
    ]
    return "\n".join(lines), failed


if __name__ == "__main__":
    root = Path(sys.argv[1])
    report, failed = summarize(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text(report)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(report)
    print(report)
    raise SystemExit(
        int(
            failed
            or os.environ.get("GROUP_RESULT") != "success"
            or os.environ.get("BUILD_RESULT") != "success"
        )
    )
