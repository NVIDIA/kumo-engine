# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "packages/kumo-engine/test/cases"
common = SimpleNamespace(**runpy.run_path(str(CASES / "common.py")))


@pytest.mark.parametrize(
    "workflow,controller,worker",
    [("kumo-engine-e2e.yml", "pool", "cases"),
     ("engine-e2e-lifecycle.yml", "lifecycle", "failed-case")])
def test_controllers_and_case_jobs_use_separate_oidc_identities(
        workflow, controller, worker):
    jobs = yaml.safe_load(
        (ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    assert jobs[controller]["environment"] == dict(name="engine-e2e-pool",
                                                   deployment=False)
    assert jobs[worker]["environment"] == dict(name="gha-kumo-workflow-tests",
                                               deployment=False)
    role = next(step["with"]["role-to-assume"]
                for step in jobs[controller]["steps"]
                if step.get("uses", "").startswith(
                    "aws-actions/configure-aws-credentials@"))
    assert role == "${{ vars.ENGINE_E2E_POOL_ROLE }}"


def test_requirements_expose_declared_hardware_without_engine_imports():
    for scale, scratch, instance in [("100M", 100, "g7e.8xlarge"),
                                     ("25B", 3200, "g7e.24xlarge")]:
        output = subprocess.check_output([
            sys.executable, "-S",
            str(CASES / "adtech_binary.py"), "--scale", scale, "--requirements"
        ], text=True)
        assert json.loads(output) == dict(scratch_gib=scratch,
                                          instance=instance)


def test_ci_checks_actual_instance_but_local_execution_ignores_it(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "case", "--scale", "100M", "--reference", "reference.json",
        "--source-sha", "a" * 40
    ])
    monkeypatch.setenv("ENGINE_E2E_DATA_S3", "s3://test")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("ENGINE_E2E_EXPECTED_INSTANCE_TYPE", "g7e.8xlarge")
    scales = {"100M": common.Scale(100, "g7e.8xlarge")}
    for actual in ("", "g7e.24xlarge"):
        monkeypatch.setenv("ENGINE_E2E_INSTANCE_TYPE", actual)
        with pytest.raises(SystemExit):
            common.parse_args("case", scales, family="adtech", **common.LIMITS)
    monkeypatch.setenv("ENGINE_E2E_INSTANCE_TYPE", "g7e.8xlarge")
    assert common.parse_args("case", scales, family="adtech",
                             **common.LIMITS).instance_type == "g7e.8xlarge"
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    monkeypatch.setenv("ENGINE_E2E_INSTANCE_TYPE", "other")
    assert common.parse_args("case", scales, family="adtech",
                             **common.LIMITS).scratch_gib == 100


def test_reference_comparison_requires_equal_known_instance_types():
    def result(instance):
        return dict(graph={}, training_table={},
                    hardware=dict(instance_type=instance))

    for current, reference in [("g7e.8xlarge", "g7e.8xlarge"), (None, None)]:
        common.compare_reference(result(current), result(reference), .25)
    for current, reference in [("g7e.8xlarge", "g7e.24xlarge"),
                               (None, "g7e.8xlarge"), ("g7e.8xlarge", None)]:
        with pytest.raises(ValueError, match="instance"):
            common.compare_reference(result(current), result(reference), .25)


@pytest.mark.parametrize("tier", ["nightly", "weekly"])
def test_runner_registers_with_repository_scope(tmp_path, tier):
    runner = ROOT / ".github/scripts/engine-e2e-runner.sh"
    source = runner.read_text()
    start, end = source.index("mode=()\n"), source.index("unset RUNNER_TOKEN")
    command = source[start:end]
    (tmp_path / "config.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    (tmp_path / "config.sh").chmod(0o755)
    result = subprocess.check_output(
        ["/bin/bash", "-c",
         command.replace("runuser -u engine-e2e -- ", "")], cwd=tmp_path,
        env=dict(os.environ, PATH="/usr/bin:/bin", TIER=tier,
                 REPOSITORY="NVIDIA/kumo-engine", LABEL="pool-type",
                 RUNNER_TOKEN="fixture-token", RUNNER_GROUP="old"),
        text=True).splitlines()
    repository_url = "https://github.com/NVIDIA/kumo-engine"
    assert result[result.index("--url") + 1] == repository_url
    assert "--runnergroup" not in result
    assert ("--ephemeral" in result) == (tier == "weekly")


@pytest.mark.parametrize("status", [0, 37])
def test_formatter_gets_a_terminal_and_preserves_its_exit_status(
        tmp_path, status):
    formatter = ROOT / "packages/diskgraph/scripts/localssd-raid0-xfs-tui.sh"
    functions = "\n".join(
        re.search(rf"^{name}\(\) \{{\n.*?^\}}", formatter.read_text(),
                  re.MULTILINE | re.DOTALL).group()
        for name in ("tty_out", "choose_disks"))
    fixture = tmp_path / "formatter.sh"
    fixture.write_text(
        "set -euo pipefail\nASSUME_YES=true\n" + functions + "\n"
        "disk_info() { printf 'fixture disk\\n'; }\n"
        "[[ $1 == --yes && $DISKGRAPH_DATA_ROOT == /mnt/localssd ]]\n"
        "mapfile -t selected < <(choose_disks /dev/fixture)\n"
        "[[ ${#selected[@]} == 1 && ${selected[0]} == /dev/fixture ]]\n"
        "exit \"$FORMATTER_EXIT\"\n")
    runner = ROOT / ".github/scripts/engine-e2e-runner.sh"
    command = next(line for line in runner.read_text().splitlines()
                   if line.startswith("DISKGRAPH_DATA_ROOT="))
    result = subprocess.run([
        "/bin/bash", "-c",
        command.replace("/run/engine-e2e-localssd.sh", str(fixture))
    ], env=dict(os.environ, PATH="/usr/bin:/bin", FORMATTER_EXIT=str(status)),
                            stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, start_new_session=True, timeout=10)
    assert result.returncode == status, result.stdout + result.stderr


@pytest.mark.parametrize("mode,status", [("failure", 1), ("report-failure", 1),
                                         ("drained", 0), ("job-exit", 7)])
def test_bootstrap_reports_its_failed_check_and_always_shuts_down(
        tmp_path, mode, status):
    source = (ROOT / ".github/scripts/engine-e2e-runner.sh").read_text()
    functions = "\n".join(
        re.findall(r"^[a-z_]+\(\) \{\n.*?^\}", source,
                   re.MULTILINE | re.DOTALL))
    trap = next(line for line in source.splitlines()
                if line.startswith("trap ") and line.endswith(" EXIT"))
    body = "boot_check=GPU\nfalse\n"
    if mode == "drained":
        body = source.split("and local SSD'\n",
                            1)[1].split("cd /opt/actions-runner", 1)[0]
    elif mode == "job-exit":
        body = source[source.index("printf 'ENGINE_E2E_INSTANCE_TYPE="):]
    commands = dict(mktemp='exec /usr/bin/mktemp "$TEST_DIR/token.XXXXXX"',
                    shutdown='printf shutdown >> "$TEST_DIR/shutdown.log"',
                    chown="exit 0", runuser="exit 7")
    for name, command in commands.items():
        (tmp_path / name).write_text("#!/bin/bash\n" + command + "\n")
        (tmp_path / name).chmod(0o755)
    (tmp_path / "aws").write_text("""#!/usr/bin/python3
import json, os, sys
from pathlib import Path
payload = json.loads(sys.argv[sys.argv.index('--payload') + 1])
with (Path(os.environ['TEST_DIR']) / 'events').open('a') as output:
    output.write(json.dumps(payload) + '\\n')
if os.environ['TEST_MODE'] == 'report-failure':
    raise SystemExit(31)
Path(sys.argv[-1]).write_text('null')
print('None')
""")
    (tmp_path / "aws").chmod(0o755)
    (tmp_path / "events").touch()
    cases = [
        dict(case="adtech_binary", scale="100M", instance="g7e.8xlarge",
             label="engine-e2e-41-2-g7e.8xlarge")
    ]
    result = subprocess.run([
        "/bin/bash", "-c", "set -euo pipefail\nbootstrapping=1\n" + functions +
        "\n" + trap + "\n" + body
    ], cwd=tmp_path, env=dict(
        os.environ, PATH=f"{tmp_path}:/usr/bin:/bin", TEST_DIR=str(tmp_path),
        TEST_MODE=mode, REPOSITORY="NVIDIA/kumo-engine", POOL_RUN_ID="41",
        POOL_ATTEMPT="2", SOURCE_SHA="a" * 40, CAPACITY_DEADLINE="2000000000",
        INSTANCE_TYPE="g7e.8xlarge", LABEL=cases[0]["label"],
        POOL_CASES=json.dumps(cases), CAPACITY_FUNCTION="fixture",
        AWS_REGION="us-west-2", actual_instance="g7e.8xlarge"),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == status, result.stdout + result.stderr
    lines = (tmp_path / "events").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert [event["kind"] for event in events
            ] == (["bootstrap"] if mode.endswith("failure") else
                  ["registration"] if mode == "drained" else [])
    if events:
        assert events[0]["cases"] == cases
    if mode.endswith("failure"):
        assert events[0]["error"] == "Bootstrap failed: GPU (exit 1)"
        assert events[0]["error"] in result.stderr
    if mode == "report-failure":
        assert "Bootstrap failure notification failed" in result.stderr
    assert (tmp_path / "shutdown.log").read_text() == "shutdown"
    assert not list(tmp_path.glob("token.*"))


@pytest.mark.parametrize("deadline,requests", [(2000000000, 2), (1, 1)])
def test_admission_preserves_bootstrap_failure_when_claim_already_exists(
        tmp_path, monkeypatch, deadline, requests):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/kumo-engine-e2e.yml").read_text())
    step = next(
        s for s in workflow["jobs"]["cases"]["steps"]
        if s.get("name") == "Admit the case before the capacity deadline")
    script = step["run"].split("\n", 1)[1].rsplit("\nPY", 1)[0]
    error = "Bootstrap failed: GPU (exit 1)"
    marker = dict(state="bootstrap_failed", instance="g7e.8xlarge",
                  deadline=deadline, error=error)
    calls = []

    def aws(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["s3api", "put-object"]:
            return subprocess.CompletedProcess(command, 1, "",
                                               "PreconditionFailed")
        assert command[1:3] == ["s3", "cp"] and "-" in command
        return subprocess.CompletedProcess(command, 0, json.dumps(marker), "")

    for key, value in dict(RUNNER_TEMP=str(tmp_path), CASE="adtech_binary",
                           SCALE="100M", GITHUB_RUN_ID="41",
                           GITHUB_RUN_ATTEMPT="2", GITHUB_SHA="a" * 40,
                           ENGINE_E2E_EXPECTED_INSTANCE_TYPE="g7e.8xlarge",
                           ENGINE_E2E_CAPACITY_DEADLINE=str(deadline),
                           RESULTS_S3="s3://fixture/results").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(subprocess, "run", aws)
    with pytest.raises(SystemExit):
        exec(compile(script, "admission", "exec"), {})
    result = json.loads((tmp_path / "e2e/result.json").read_text())
    assert result["phase"] == "bootstrap"
    assert result["error"] == error
    assert len(calls) == requests
