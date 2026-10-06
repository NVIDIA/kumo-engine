# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from scripts.summarize_gitleaks_report import (
    build_report,
    render_markdown,
    write_reports,
)


def test_persistent_report_excludes_detected_and_commit_metadata(
        tmp_path: Path) -> None:
    records = [{
        "RuleID": "example-token",
        "File": "src/config.py",
        "StartLine": 7,
        "EndLine": 7,
        "Commit": "abc123",
        "Secret": "sensitive-value",
        "Match": "TOKEN=sensitive-value",
        "Author": "Example Person",
        "Email": "person@example.test",
        "Message": "temporary credential",
    }]

    raw_path = tmp_path / "raw.json"
    json_path = tmp_path / "sanitized.json"
    markdown_path = tmp_path / "sanitized.md"
    raw_path.write_text(json.dumps(records))
    write_reports(raw_path, json_path, markdown_path)
    report = json.loads(json_path.read_text())
    outputs = json_path.read_text() + markdown_path.read_text()

    assert "sensitive-value" not in outputs
    assert "abc123" not in outputs
    assert "Example Person" not in outputs
    assert "person@example.test" not in outputs
    assert "temporary credential" not in outputs
    assert "findings" not in report
    assert report["schema_version"] == 2
    assert report["totals"] == {
        "findings": 1,
        "commits": 1,
        "files": 1,
    }


def test_report_groups_rules_and_files() -> None:
    records = [{
        "RuleID": "example-token",
        "File": "src/config.py",
        "StartLine": line,
        "Commit": commit,
    } for line, commit in [(7, "abc123"), (8, "def456")]]

    report = build_report(json.dumps(records).encode())

    assert report["rules"] == [{
        "rule_id": "example-token",
        "findings": 2,
        "files": 1,
        "commits": 2,
    }]
    assert report["files"] == [{
        "path": "src/config.py",
        "findings": 2,
    }]


def test_empty_report_has_clean_status_and_no_remediation_steps() -> None:
    report = build_report(b"[]")
    markdown = render_markdown(report, "sanitized.json")

    assert report["totals"] == {"findings": 0, "commits": 0, "files": 0}
    assert "No candidate credential material found" in markdown
    assert "Required handling" not in markdown
    assert "Revoke or rotate" not in markdown
