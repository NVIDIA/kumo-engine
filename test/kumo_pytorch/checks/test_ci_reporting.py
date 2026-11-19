# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reporting must retain evidence and never change command outcomes."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from test.kumo_pytorch import ci_reporting as reporting

import pytest

SCRIPT = str(Path(reporting.__file__).resolve())


def test_fitting_and_training_summaries_name_the_failed_requirement():
    rows = reporting.json_checks(
        {
            "status": "failed",
            "comparison": {
                "status": "failed",
                "reason": "Unavailable anonymous fitting memory",
            },
            "runs": [{"status": "failed", "error": "Worker timed out"}],
            "repeats": [
                {"checks": {"every_parameter_updated": False, "loss_decreased": True}}
            ],
        }
    )
    assert any(r["reason"] == "Unavailable anonymous fitting memory" for r in rows)
    assert any(r["reason"] == "Worker timed out" for r in rows)
    assert any(
        r["check"].endswith("every_parameter_updated") and r["status"] == "failed"
        for r in rows
    )


def test_artifact_upload_failure_does_not_claim_retained_logs(tmp_path):
    text = reporting.summary(tmp_path, "Example", "", "failure (artifact upload)")
    assert (
        "failure (artifact upload)" in text
        and "Diagnostics artifact unavailable" in text
    )
    assert "are in the linked artifact" not in text


def test_cancellation_escalates_before_the_next_heartbeat(tmp_path):
    marker = tmp_path / "started"
    command = f"import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path({str(marker)!r}).touch(); time.sleep(60)"
    process = subprocess.Popen(
        [
            sys.executable,
            SCRIPT,
            "run",
            "--title",
            "Signal escalation",
            "--",
            sys.executable,
            "-c",
            command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "KUMO_CI_REPORTS": ""},
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=12)
        assert process.returncode == 143 and time.monotonic() - started < 10
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=12)


def execute(tmp_path, command, *, input=None):
    return subprocess.run(
        [sys.executable, SCRIPT, "run", "--title", "Example check", "--", *command],
        input=input,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "KUMO_CI_REPORTS": str(tmp_path / "report.*"),
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        },
        cwd=tmp_path,
        timeout=20,
    )


def test_nested_lifecycle_reason_replaces_opaque_failure_and_keeps_evidence(tmp_path):
    report = {
        "status": "failed",
        "error": "Lifecycle checks failed; see individual results",
        "lifecycle": {
            "status": "failed",
            "checks": {
                "inference": {"status": "passed", "model_sha256": "PRIVATE_DETAILS"},
                "finetuning": {"status": "passed"},
                "mid_epoch_resume": {
                    "status": "failed",
                    "error": "this state was saved for a different task",
                    "traceback": "FULL_TRACEBACK",
                },
            },
        },
    }
    result = execute(
        tmp_path,
        [
            sys.executable,
            "-c",
            f"import json; from pathlib import Path; Path('report.json').write_text({json.dumps(json.dumps(report))}); print('x'*50000); raise SystemExit(7)",
        ],
    )
    assert result.returncode == 7
    assert "mid epoch resume" in result.stdout
    assert "Observed: this state was saved for a different task" in result.stdout
    assert len(result.stdout) < 2000
    assert (
        "PRIVATE_DETAILS" not in result.stdout and "FULL_TRACEBACK" not in result.stdout
    )
    saved = next((tmp_path / "pytorch-ci-logs").glob("*/001-report.json"))
    assert json.loads(saved.read_text()) == report
    assert saved.with_name("console.log").read_text() == "x" * 50000 + "\n"


def test_junit_counts_failures_errors_and_skips_without_dumping_tracebacks(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text("""<testsuites><testsuite>
      <testcase classname="a" name="pass"/>
      <testcase classname="a" name="failure"><failure message="expected 4, got 3">LONG TRACEBACK</failure></testcase>
      <testcase classname="a" name="setup"><error message="missing input"/></testcase>
      <testcase classname="a" name="skip"><skipped message="unsupported"/></testcase>
    </testsuite></testsuites>""")
    rows, counts = reporting.read_report(path)
    assert counts == {"passed": 1, "failed": 2, "skipped": 1}
    assert {row["reason"] for row in rows} == {
        "expected 4, got 3",
        "missing input",
        "unsupported",
    }


def test_failure_summary_explains_fingerprints_and_keeps_raw_evidence(tmp_path):
    raw = (
        "AssertionError: assert identities equal\nOmitting 6 identical items, use -vv to show\n"
        "Differing items: {'anchors': 'sha256:aaaa'} != {'anchors': 'sha256:bbbb'}"
    )
    xml = (
        '<testsuites><testsuite><testcase classname="tests.test_anchor_tables.TestIdentity" '
        'name="test_a_task_carries_its_fingerprints_and_output_schema">'
        f'<failure message="{raw.replace(chr(10), "&#10;")}">COMPLETE TRACEBACK</failure>'
        "</testcase></testsuite></testsuites>"
    )
    result = execute(
        tmp_path,
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path('report.xml').write_text({xml!r}); raise SystemExit(1)",
        ],
    )
    assert result.returncode == 1
    assert "Repeated materialization preserves task identity" in result.stdout
    assert "Expected: Identical inputs produce identical fingerprints" in result.stdout
    assert "Observed: Fingerprint mismatch in anchors" in result.stdout
    for opaque in ("sha256:", "Omitting", "test_a_task_carries", "COMPLETE TRACEBACK"):
        assert opaque not in result.stdout
    retained = next((tmp_path / "pytorch-ci-logs").glob("*/001-report.xml"))
    assert retained.read_text() == xml
    markdown = reporting.summary(
        tmp_path / "pytorch-ci-logs", "Package tests", "", "failure"
    )
    assert "| What was tested | Expected | Observed |" in markdown
    assert "<summary>Exact test IDs and documentation</summary>" in markdown
    assert "test_a_task_carries_its_fingerprints_and_output_schema" in markdown
    assert "sha256:" not in markdown


def test_split_mismatch_explains_missing_metadata_and_keeps_variant(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text("""<testsuites><testsuite>
      <testcase classname="tests.test_inference_lifecycle" name="test_inference_preserves_batches_after_index_rebuild[pragma-inspectable]">
        <failure message="AssertionError: metadata.splits[0]&#10;assert None == 'train'"/>
      </testcase>
    </testsuite></testsuites>""")
    rows, counts = reporting.read_report(path)
    assert counts == {"failed": 1}
    assert (
        rows[0]["title"]
        == "Rebuilding the index preserves inference results and metadata."
    )
    assert rows[0]["variant"] == "pragma-inspectable"
    assert "meaningful split labels" in rows[0]["expected"]
    assert (
        rows[0]["reason"]
        == "Split metadata differs at position 0: missing versus 'train'."
    )


def test_descriptions_are_read_without_importing_tests_and_fallback_is_honest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(reporting, "REPOSITORY", tmp_path)
    (tmp_path / "example.py").write_text('''raise RuntimeError("MUST NOT IMPORT")
class TestCases:
    async def test_example(self):
        """Readable purpose.

        Expected: The saved values remain unchanged.

        Evidence: docs/contract.md#reload
        """
''')
    detail = reporting.test_description("example.TestCases", "test_example[variant]")
    assert detail == {
        "title": "Readable purpose.",
        "expected": "The saved values remain unchanged.",
        "evidence": "docs/contract.md#reload",
        "variant": "variant",
    }
    fallback = reporting.test_description(
        "missing_module", "test_handles_empty_input[packed]"
    )
    assert fallback == {"title": "Handles empty input", "variant": "packed"}
    assert "expected" not in fallback
    # Missing source never changes a failure into a reporting error.
    assert reporting.source_descriptions(tmp_path / "missing.py") == {}


def test_observations_do_not_invent_identity_failures_or_hide_ambiguity():
    unrelated = "RuntimeError: connection refused"
    assert reporting.observation(unrelated) == unrelated
    non_hash = "AssertionError: Differing items: {'count': 3} != {'count': 4}"
    assert (
        reporting.observation(non_hash)
        == "Compared mappings differ: {'count': 3} != {'count': 4}"
    )
    ambiguous = (
        "AssertionError: CONTRACT AMBIGUITY A01: two readings; see docs/contract.md."
    )
    assert reporting.observation(ambiguous).startswith("CONTRACT AMBIGUITY A01")
    long_ambiguity = (
        "AssertionError: CONTRACT AMBIGUITY A01: "
        + "explanation " * 50
        + "Observed ValueError: event token metadata does not match stored fields"
    )
    assert reporting.observation(long_ambiguity) == (
        "CONTRACT AMBIGUITY A01: ValueError: event token metadata does not match stored fields"
    )
    for diagnostic in (
        "ArrowInvalid: Integer value 128 not in range: -128 to 127",
        "AssertionError: metadata.sample_ids[0]\nassert 'wrong' == 'user-1'",
        "AssertionError: Regex pattern did not match: this state was saved for a different task",
    ):
        assert reporting.observation(diagnostic) == reporting.brief(diagnostic)


def test_mapping_difference_is_retained_after_pytest_omission_advice():
    diagnostic = (
        "assert {'field_0': 0, ...} == {'field_0': 0, ...}\n"
        "  Omitting 99 identical items, use -vv to show\n"
        "  Differing items:\n"
        "  {'field_99': 99} != {'field_99': -1}\n"
        "  Use -v to get more diff"
    )
    assert reporting.observation(diagnostic) == (
        "Compared mappings differ: {'field_99': 99} != {'field_99': -1}"
    )


def test_empty_field_diagnostic_preserves_command_exit_and_report(tmp_path):
    xml = """<testsuites><testsuite><testcase name="test_missing_lengths">
      <failure message="ValueError: post-fit text token lengths missing for "/>
    </testcase></testsuite></testsuites>"""
    result = execute(
        tmp_path,
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path('report.xml').write_text({xml!r}); raise SystemExit(7)",
        ],
    )
    assert result.returncode == 7
    assert "token-length metadata (field not reported)" in result.stdout
    saved = next((tmp_path / "pytorch-ci-logs").glob("*/step.json"))
    assert json.loads(saved.read_text())["exit_code"] == 7


def test_junit_refusal_is_separate_and_cannot_hide_failed_or_skipped_case(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text("""<testsuites><testsuite>
      <testcase name="refused"><properties><property name="capability_refusals" value="[]"/></properties></testcase>
      <testcase name="later failure"><properties><property name="capability_refusals" value="[]"/></properties><failure>AssertionError: wrong result</failure></testcase>
      <testcase name="skip"><properties><property name="capability_refusals" value="[]"/></properties><skipped/></testcase>
    </testsuite></testsuites>""")
    rows, counts = reporting.read_report(path)
    assert counts == {"refused (accepted)": 1, "failed": 1, "skipped": 1}
    assert rows[1]["reason"] == "AssertionError: wrong result"


def test_benchmark_incompleteness_reports_experiment_and_metric_reasons():
    rows = reporting.json_checks(
        {
            "status": "incomplete",
            "experiments": {
                "cold_and_warm_caches": {
                    "status": "incomplete",
                    "reason": "Files changed between epochs",
                }
            },
            "metrics": {
                "tokens_per_second": {
                    "status": "incomplete",
                    "cases": [
                        {
                            "status": "missing",
                            "reason": "Required measurement was not collected",
                        }
                    ],
                }
            },
            "measurements": {"huge": list(range(1000))},
        }
    )
    assert any(
        r["check"].endswith("cold_and_warm_caches")
        and r["reason"] == "Files changed between epochs"
        for r in rows
    )
    assert any(
        "tokens_per_second" in r["check"]
        and r["reason"] == "Required measurement was not collected"
        for r in rows
    )
    assert len(rows) == 4


def test_download_errors_name_the_file_and_generic_reports_use_log_reason(tmp_path):
    rows = reporting.json_checks(
        {
            "status": "failed",
            "results": [
                {
                    "path": "transactions.parquet",
                    "status": "failed",
                    "error": "AccessDenied",
                }
            ],
        }
    )
    assert rows[1]["check"] == "results / transactions.parquet"
    assert rows[1]["reason"] == "AccessDenied"
    result = execute(
        tmp_path,
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('report.json').write_text('{\"status\": \"failed\"}'); print('PermissionError: source cannot be read'); raise SystemExit(4)",
        ],
    )
    assert (
        result.returncode == 4
        and "PermissionError: source cannot be read" in result.stdout
    )


@pytest.mark.parametrize("code", [0, 1, 5, 127])
def test_exit_codes_preserved_even_without_reports(tmp_path, code):
    result = execute(
        tmp_path,
        [
            sys.executable,
            "-c",
            f"print('Reason: deliberate failure'); raise SystemExit({code})",
        ],
    )
    assert result.returncode == code
    assert ("::error::" in result.stdout) == bool(code)


def test_bash_input_environment_outputs_and_pipefail_are_preserved(tmp_path):
    result = execute(
        tmp_path,
        ["bash", "-e", "-o", "pipefail"],
        input='echo "answer=42" >> "$GITHUB_OUTPUT"\npwd\nfalse | cat\necho unreachable\n',
    )
    assert result.returncode == 1
    assert (tmp_path / "outputs").read_text() == "answer=42\n"
    log = next((tmp_path / "pytorch-ci-logs").glob("*/console.log")).read_text()
    assert log.strip() == str(tmp_path.resolve())


def test_malformed_reports_and_missing_executable_do_not_replace_exit_code(tmp_path):
    result = execute(
        tmp_path,
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('report.json').write_text('{'); raise SystemExit(9)",
        ],
    )
    assert result.returncode == 9 and "Cannot read report.json" in result.stdout
    missing = execute(tmp_path, ["/missing/test-executable"])
    assert missing.returncode == 127 and "FileNotFoundError" in missing.stdout


def test_summary_is_bounded_escaped_and_distinguishes_accepted_refusals(tmp_path):
    checks = [
        {"check": f"case-{i}", "status": "failed", "reason": "<script> | bad\nvalue"}
        for i in range(30)
    ]
    checks += reporting.json_checks(
        {
            "status": "passed",
            "outcome": "refused",
            "refusal": {"message": "unsupported capability"},
        }
    )
    directory = tmp_path / "step"
    directory.mkdir()
    (directory / "step.json").write_text(
        json.dumps(
            {
                "title": "Training",
                "status": "failed",
                "seconds": 1,
                "tests": {"passed": 7, "failed": 30, "skipped": 2},
                "checks": checks,
                "notes": [],
            }
        )
    )
    text = reporting.summary(
        tmp_path, "Suite", "https://github.com/example/actions/artifacts/123", "failure"
    )
    assert "Showing 12 of 30" in text and "2 skipped" in text
    assert "<script>" not in text and "&#124;" in text
    assert "Accepted capability refusals" in text and "artifacts/123" in text
    assert len(text) < 4000


def test_cancellation_keeps_partial_output_and_returns_signal_status(tmp_path):
    process = subprocess.Popen(
        [
            sys.executable,
            SCRIPT,
            "run",
            "--title",
            "Interrupted",
            "--",
            sys.executable,
            "-u",
            "-c",
            "import time; print('partial output', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "KUMO_CI_REPORTS": ""},
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            logs = list((tmp_path / "pytorch-ci-logs").glob("*/console.log"))
            if logs and "partial output" in logs[0].read_text():
                break
            time.sleep(0.05)
        else:
            pytest.fail("Child did not start")
        process.send_signal(signal.SIGTERM)
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == 143 and "CANCELLED" in stdout
        assert "partial output" in logs[0].read_text()
        assert (
            json.loads(logs[0].with_name("step.json").read_text())["status"]
            == "cancelled"
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_cancellation_kills_workers_when_their_parent_exits_first(tmp_path):
    marker = tmp_path / "worker-alive"
    child = (
        "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=Path(%r);\nwhile True:\n p.write_text(str(time.time_ns())); time.sleep(.02)"
        % str(marker)
    )
    parent = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(60)"
        % child
    )
    process = subprocess.Popen(
        [
            sys.executable,
            SCRIPT,
            "run",
            "--title",
            "Worker cancellation",
            "--",
            sys.executable,
            "-c",
            parent,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "KUMO_CI_REPORTS": ""},
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 143
        after = marker.read_text()
        time.sleep(0.1)
        assert marker.read_text() == after
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=10)
