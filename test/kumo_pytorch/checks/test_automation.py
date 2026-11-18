# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual gate shell decisions and complete nightly job accounting."""

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from test.kumo_pytorch import automation

import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"
GATE = (WORKFLOWS / "kumo-pytorch-gate.yml").read_text()


def shell_steps():
    # Extract literal shell blocks, not a second implementation of gate policy.
    return [
        textwrap.dedent(block)
        for block in re.findall(
            r"^        run: \|\n((?:^          .*\n|^\n)+)", GATE, re.MULTILINE
        )
    ]


@pytest.mark.parametrize("job", ["WHEEL", "CPU", "DISKGRAPH", "AML", "TOOLING"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", ""])
def test_affected_gate_rejects_every_unsuccessful_dependency(job, result):
    env = {
        f"{name}_RESULT": "success"
        for name in ("CHANGES", "WHEEL", "CPU", "DISKGRAPH", "AML", "TOOLING")
    }
    env.update(AFFECTED="true", **{f"{job}_RESULT": result})
    assert (
        subprocess.run(
            ["bash", "-c", shell_steps()[-1]],
            env={**os.environ, **env},
            capture_output=True,
            check=False,
        ).returncode
        == 1
    )


@pytest.mark.parametrize(
    "affected,changes,expected",
    [
        ("true", "success", 0),
        ("false", "success", 0),
        ("false", "failure", 1),
        ("", "success", 1),
    ],
)
def test_gate_requires_valid_detection_or_all_checks(affected, changes, expected):
    env = {
        f"{name}_RESULT": "success" if affected == "true" else "skipped"
        for name in ("WHEEL", "CPU", "DISKGRAPH", "AML", "TOOLING")
    }
    env.update(AFFECTED=affected, CHANGES_RESULT=changes)
    assert (
        subprocess.run(
            ["bash", "-c", shell_steps()[-1]],
            env={**os.environ, **env},
            capture_output=True,
            check=False,
        ).returncode
        == expected
    )


@pytest.mark.parametrize(
    "path,expected",
    [
        ("test/kumo_pytorch/required-tests.json", "true"),
        ("test/kumo_pytorch/model_checks/test_aml.py", "true"),
        (".github/workflows/_kumo-pytorch-aml-checks.yml", "true"),
        (".github/workflows/kumo-pytorch-t2b.yml", "true"),
        ("packages/experimental/kumo-pytorch/src/kumo/pytorch/task.py", "true"),
        ("packages/diskgraph/src/lib.rs", "true"),
        ("setup.py", "true"),
        (".github/actions/rust-toolchain/action.yml", "true"),
        (".github/actions/apt-install/action.yml", "true"),
        ("pyproject.toml", "true"),
        ("uv.lock", "true"),
        ("docs/unrelated.md", "false"),
        ("test/other.py", "false"),
    ],
)
def test_changed_file_detection_uses_the_workflow(path, expected, tmp_path):
    fake = tmp_path / "gh"
    fake.write_text(
        '#!/bin/bash\nif [[ "$*" == *"/files"* ]]; then printf "%s\\n" "$FILES"; else echo 1; fi\n'
    )
    fake.chmod(0o755)
    output = tmp_path / "output"
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        FILES=path,
        GITHUB_OUTPUT=str(output),
        EVENT_NAME="pull_request",
        GITHUB_REPOSITORY="NVIDIA/kumo-engine",
        PR_NUMBER="353",
    )
    subprocess.run(
        ["bash", "-c", shell_steps()[0]], env=env, check=True, capture_output=True
    )
    assert output.read_text().strip() == f"affected={expected}"


def results():
    return {name: {"result": "success"} for name in automation.NIGHTLY_JOBS}


def evidence(jobs=None, **kwargs):
    return automation.nightly_status(
        results() if jobs is None else jobs,
        run_id=42,
        attempt=1,
        sha="abc",
        event="schedule",
        ref="refs/heads/main",
        **kwargs,
    )


def scheduled():
    return {
        "id": 42,
        "run_attempt": 1,
        "head_sha": "abc",
        "event": "schedule",
        "head_branch": "main",
        "path": ".github/workflows/kumo-pytorch-t2b.yml",
        "status": "completed",
        "conclusion": "success",
        "html_url": "https://example.test/run/42",
    }


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "failure", "skipped", "cancelled"]
)
def test_nightly_never_hides_missing_or_unsuccessful_jobs(fault):
    jobs = results()
    if fault == "missing":
        jobs.pop("benchmark-gpu")
    elif fault == "extra":
        jobs["unregistered"] = {"result": "success"}
    else:
        jobs["benchmark-gpu"]["result"] = fault
    assert evidence(jobs)["status"] == "failed"


def test_nightly_status_lists_every_workflow_job_and_dependency():
    text = (WORKFLOWS / "kumo-pytorch-t2b.yml").read_text().split("\njobs:\n", 1)[1]
    jobs = set(re.findall(r"^  ([\w-]+):$", text, re.MULTILINE))
    assert jobs - {"nightly-result"} == automation.NIGHTLY_JOBS
    final = text.split("  nightly-result:\n", 1)[1]
    needs = re.search(r"    needs: \[(.*)\]", final)[1].split(", ")
    assert set(needs) == automation.NIGHTLY_JOBS
    assert "always()" in final and "toJSON(needs)" in final
    assert (
        "--required-coverage=cpu"
        in (WORKFLOWS / "_kumo-pytorch-aml-checks.yml").read_text()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "workflow_dispatch"),
        ("head_branch", "feature"),
        ("head_sha", "other"),
        ("id", 43),
        ("run_attempt", 2),
        ("status", "in_progress"),
        ("path", "other.yml"),
    ],
)
def test_schedule_proof_rejects_manual_runs_and_stale_evidence(field, value):
    with pytest.raises(ValueError):
        automation.verify_schedule({**scheduled(), field: value}, evidence())


def test_schedule_trigger_is_distinct_from_test_success():
    assert automation.verify_schedule(scheduled(), evidence())["tests"] == "passed"
    jobs = results()
    jobs["aml-model-checks"]["result"] = "failure"
    proof = automation.verify_schedule(
        {**scheduled(), "conclusion": "failure"}, evidence(jobs)
    )
    assert proof["scheduled_trigger"] == "verified" and proof["tests"] == "failed"
    assert proof["required_jobs_executed"]
    jobs["benchmark-gpu"]["result"] = "skipped"
    proof = automation.verify_schedule(
        {**scheduled(), "conclusion": "failure"}, evidence(jobs)
    )
    assert not proof["required_jobs_executed"]


def test_failed_nightly_writes_status_before_exiting(tmp_path):
    path = tmp_path / "status.json"
    env = dict(
        os.environ,
        JOB_RESULTS=json.dumps({}),
        GITHUB_RUN_ID="42",
        GITHUB_RUN_ATTEMPT="1",
        GITHUB_SHA="abc",
        GITHUB_EVENT_NAME="schedule",
        GITHUB_REF="refs/heads/main",
        GITHUB_STEP_SUMMARY=str(tmp_path / "summary.md"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "test.kumo_pytorch.automation",
            "record",
            "--output",
            str(path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1 and json.loads(path.read_text())["missing"]
