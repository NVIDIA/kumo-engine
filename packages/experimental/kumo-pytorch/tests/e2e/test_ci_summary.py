# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from tests.e2e.ci_summary import render


def test_summary_caps_failures_and_escapes_assertions(tmp_path):
    case = """<testcase name="test_row"><properties>
      <property name="nodeid" value="test_row.py::test_row[packed]"/>
      <property name="purpose" value="Match source labels"/></properties>
      <failure message="expected 2, observed &lt;3&gt;"/></testcase>"""
    cases = case + "".join(case.replace("packed", f"row-{i}") for i in range(12))
    (tmp_path / "junit.xml").write_text("<testsuite>" + cases + "</testsuite>")
    summary = render(tmp_path, "https://example.org/attempt", "failure")
    assert summary.count("Match source labels") == 12
    assert "test_row.py::test_row[packed]" in summary
    assert "expected 2, observed &lt;3&gt;" in summary
    assert "12 of 13" in summary and "https://example.org/attempt" in summary


def test_summary_falls_back_to_current_log_and_reads_worker_reports(tmp_path):
    (tmp_path / "pytest.log").write_text("setup stopped\n")
    assert "setup stopped" in render(tmp_path, "", "failure")
    report = tmp_path / "timings.json"
    report.write_text(json.dumps({"workload": "L01", "workers": [{"returncode": 23}]}))
    summary = render(tmp_path, "", "failure", [report])
    assert "L01" in summary and "23" in summary and "setup stopped" not in summary
    report.write_text("[]")
    summary = render(tmp_path, "", "failure", [report])
    assert "Unreadable evidence" in summary and "setup stopped" in summary
