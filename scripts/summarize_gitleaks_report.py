#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a sanitized summary from a Gitleaks JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def build_report(raw_report: bytes,
                 generated_at: str | None = None) -> dict[str, Any]:
    records = json.loads(raw_report)
    if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records):
        raise ValueError("Gitleaks report must be a JSON array of objects")

    rule_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    rule_files: dict[str, set[str]] = defaultdict(set)
    rule_commits: dict[str, set[str]] = defaultdict(set)
    commits: set[str] = set()
    for record in records:
        rule_id = str(record.get("RuleID", "unknown"))
        path = str(record.get("File", "unknown"))
        commit = str(record.get("Commit", ""))
        rule_counts[rule_id] += 1
        file_counts[path] += 1
        rule_files[rule_id].add(path)
        if commit:
            commits.add(commit)
            rule_commits[rule_id].add(commit)

    return {
        "schema_version":
        2,
        "generated_at":
        generated_at or datetime.now(timezone.utc).isoformat(),
        "source_sha256":
        hashlib.sha256(raw_report).hexdigest(),
        "totals": {
            "findings": len(records),
            "commits": len(commits),
            "files": len(file_counts),
        },
        "rules": [{
            "rule_id": rule_id,
            "findings": count,
            "files": len(rule_files[rule_id]),
            "commits": len(rule_commits[rule_id]),
        } for rule_id, count in sorted(rule_counts.items())],
        "files": [{
            "path": path,
            "findings": count,
        } for path, count in sorted(file_counts.items(), key=lambda item:
                                    (-item[1], item[0]))],
    }


def _markdown_text(value: Any) -> str:
    return (str(value).replace("\\", "\\\\").replace("|", "\\|").replace(
        "<", "&lt;").replace(">", "&gt;").replace("`", "\\`"))


def render_markdown(report: dict[str, Any], json_link: str) -> str:
    totals = report["totals"]
    has_findings = totals["findings"] > 0
    status = ("Requires credential review and history remediation."
              if has_findings else "No candidate credential material found.")
    lines = [
        "# Gitleaks history scan",
        "",
        f"**Status: {status}**",
        "",
        f"Generated: `{_markdown_text(report['generated_at'])}`  ",
        f"Source SHA-256: `{_markdown_text(report['source_sha256'])}`",
        "",
        ("The scan found candidate credential material; it does not establish "
         "whether any value is valid or still active." if has_findings else
         "The scan found no candidate credential material."),
        "The source report may contain detected text and is intentionally "
        "excluded from persistent task storage.",
        "",
        "## Summary",
        "",
        f"- Findings: {totals['findings']}",
        f"- Commits: {totals['commits']}",
        f"- Files: {totals['files']}",
        f"- Aggregate report: [sanitized JSON]({json_link})",
        "",
        "## Rules",
        "",
        "| Rule | Findings | Files | Commits |",
        "| --- | ---: | ---: | ---: |",
    ]
    for rule in report["rules"]:
        lines.append(
            f"| `{_markdown_text(rule['rule_id'])}` | {rule['findings']} | "
            f"{rule['files']} | {rule['commits']} |")

    lines.extend([
        "",
        "## Highest-count paths",
        "",
        "| Path | Findings |",
        "| --- | ---: |",
    ])
    for item in report["files"][:25]:
        lines.append(
            f"| `{_markdown_text(item['path'])}` | {item['findings']} |")

    if has_findings:
        lines.extend([
            "",
            "## Required handling",
            "",
            "1. Review each candidate without copying the detected value into "
            "tickets, chat, or pull requests.",
            "2. Revoke or rotate every credential that may have been usable.",
            "3. Classify false positives and add only narrow, documented "
            "scanner exceptions.",
            "4. Rewrite affected Git history only after rotation decisions "
            "are recorded.",
            "5. Rerun the history scan and require zero unexplained findings "
            "before public release.",
            "",
        ])
    else:
        lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                     dir=path.parent, delete=False) as file:
        file.write(content)
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def write_reports(raw_path: Path, json_path: Path, markdown_path: Path) -> None:
    raw_report = raw_path.read_bytes()
    report = build_report(raw_report)
    relative_json = os.path.relpath(json_path, start=markdown_path.parent)
    _atomic_write(json_path,
                  json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_markdown(report, relative_json))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_report", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args(argv)

    raw_path = args.raw_report.resolve()
    output_paths = {
        args.json_output.resolve(),
        args.markdown_output.resolve(),
    }
    if raw_path in output_paths:
        parser.error("raw input and sanitized output paths must differ")

    write_reports(raw_path, args.json_output, args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
