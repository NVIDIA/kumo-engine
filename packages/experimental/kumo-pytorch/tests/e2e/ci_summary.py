# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the current Actions job's pytest, stage and worker evidence."""

import json
import os
import sys
from collections import deque
from glob import iglob
from html import escape
from pathlib import Path
from xml.etree import ElementTree


def _failures(path):
    if path.suffix == ".xml":
        for case in ElementTree.parse(path).iter("testcase"):
            issues = case.findall("error") + case.findall("failure")
            if issues:
                props = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
                node = props.get("nodeid") or f"{case.get('classname')}::{case.get('name')}"
                purpose = props.get("purpose") or case.get("name", "").split("[")[0].removeprefix(
                    "test_"
                ).replace("_", " ")
                for issue in issues:
                    yield node, purpose, issue.get("message") or issue.text or issue.tag
        return
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or not isinstance(report.get("workers", []), list):
        raise ValueError("expected a report object with an optional worker list")
    if report.get("active_stage") or report.get("error"):
        yield (
            report.get("nodeid", path.name),
            "Correctness stage",
            str(report.get("error") or report["active_stage"]),
        )
    for worker in report.get("workers", ()):
        if not isinstance(worker, dict):
            raise ValueError("expected a worker object")
        if worker.get("complete") is False or worker.get("returncode"):
            worker_id = worker.get("results") or worker.get("worker")
            yield (
                f"{report.get('workload', path.name)} [{worker_id}]",
                "Worker completion",
                str(worker),
            )
    if report.get("complete") is False:
        yield (
            path.name,
            "Campaign completion",
            "The campaign ended before all expected work completed.",
        )


def render(directory: Path, artifact_url: str, status: str, reports=()) -> str:
    lines = [f"### Package diagnostics · {status}"]
    if artifact_url:
        lines.append(f"[Download this attempt’s evidence]({artifact_url})")
    paths = set(directory.glob("*.xml")) | set(directory.glob("stages-*.json")) | set(reports)
    seen = set()
    for path in sorted(paths, key=lambda p: (p.suffix != ".xml", p)):
        try:
            for node, purpose, detail in _failures(path):
                if node in seen:
                    continue
                seen.add(node)
                if len(seen) <= 12:
                    lines.append(
                        f"<b>{escape(purpose)}</b> <code>{escape(node)}</code>\n"
                        f"<pre>{escape(detail[:1600])}</pre>"
                    )
        except (OSError, ValueError, ElementTree.ParseError) as exc:
            lines.append(
                f"Unreadable evidence <code>{escape(path.name)}: {escape(str(exc))}</code>"
            )
    if len(seen) > 12:
        lines.append(f"Showing 12 of {len(seen)} failures. The artifact contains the full output.")
    if status != "success" and not seen:
        for log in sorted(directory.glob("*.log")):
            with log.open(errors="replace") as stream:
                tail = "".join(deque(stream, maxlen=12))[-2400:]
            lines.append(f"<code>{escape(log.name)}</code>\n<pre>{escape(tail)}</pre>")
    return "\n\n".join(lines)


if __name__ == "__main__":
    reports = (
        Path(path)
        for pattern in os.environ.get("REPORT_FILES", "").splitlines()
        if pattern.endswith((".xml", "/timings.json", "/campaign.json"))
        for path in iglob(pattern, recursive=True)
    )
    print(render(Path(sys.argv[1]), os.environ["ARTIFACT_URL"], os.environ["JOB_STATUS"], reports))
