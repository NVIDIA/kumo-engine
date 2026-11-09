# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re
import runpy
import select
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "packages/kumo-engine/test/cases"
common = SimpleNamespace(**runpy.run_path(str(CASES / "common.py")))


@pytest.mark.parametrize("tier", ["nightly", "weekly"])
def test_new_dispatch_preserves_reference_generation(tier):
    workflow = yaml.safe_load(
        (ROOT / f".github/workflows/kumo-engine-{tier}.yml").read_text())
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["concurrency"].get("queue") == "max"


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
def test_formatter_download_retries_without_rerunning_it(tmp_path, status):
    formatter = ROOT / "packages/diskgraph/scripts/localssd-raid0-xfs-tui.sh"
    functions = "\n".join(
        re.search(rf"^{name}\(\) \{{\n.*?^\}}", formatter.read_text(),
                  re.MULTILINE | re.DOTALL).group()
        for name in ("tty_out", "choose_disks"))
    fixture = tmp_path / "formatter.sh"
    content = ("set -euo pipefail\nASSUME_YES=true\n" + functions + "\n"
               "printf formatter >> \"$FORMATTER_LOG\"\n"
               "disk_info() { printf 'fixture disk\\n'; }\n"
               "[[ $1 == --yes && $DISKGRAPH_DATA_ROOT == /mnt/localssd ]]\n"
               "mapfile -t selected < <(choose_disks /dev/fixture)\n"
               "[[ ${#selected[@]} == 1 && ${selected[0]} == /dev/fixture ]]\n"
               "exit \"$FORMATTER_EXIT\"\n")
    runner = ROOT / ".github/scripts/engine-e2e-runner.sh"
    command = runner.read_text().split("boot_check='local SSD setup'\n",
                                       1)[1].split("mountpoint -q", 1)[0]
    command = command.replace(
        "https://raw.githubusercontent.com/$REPOSITORY/$SOURCE_SHA/packages/diskgraph/scripts/localssd-raid0-xfs-tui.sh",
        fixture.as_uri()).replace("/run/engine-e2e-localssd.sh",
                                  str(tmp_path / "downloaded.sh"))
    log = tmp_path / "formatter.log"
    result = subprocess.Popen(
        ["/bin/bash", "-c", "set -euo pipefail\n" + command],
        env=dict(os.environ, PATH="/usr/bin:/bin", FORMATTER_EXIT=str(status),
                 FORMATTER_LOG=str(log)), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True)
    try:
        assert select.select([result.stderr], [], [], 10)[0]
        first_error = result.stderr.readline()
        fixture.write_text(content)
        stdout, stderr = result.communicate(timeout=10)
    finally:
        if result.poll() is None:
            result.kill()
            result.wait()
    assert first_error
    assert result.returncode == status, stdout + first_error + stderr
    assert log.read_text() == "formatter"


@pytest.mark.parametrize("mode,status,attempts,reports",
                         [("failure", 1, 0, 1), ("report-failure", 1, 0, 3),
                          ("report-transport-recovery", 1, 0, 2),
                          ("report-function-recovery", 1, 0, 3),
                          ("report-deadline", 1, 0, 1),
                          ("report-expired", 1, 0, 0),
                          ("report-invalid-deadline", 1, 0, 0),
                          ("drained", 0, 1, 0), ("job-exit", 7, 1, 0),
                          ("registration-transport", 31, 3, 1),
                          ("registration-function", 1, 3, 1),
                          ("configuration", 7, 1, 1),
                          ("transport-recovery", 0, 3, 0),
                          ("function-recovery", 0, 3, 0),
                          ("drained-recovery", 0, 2, 0),
                          ("registration-deadline", 31, 1, 1)])
def test_bootstrap_reports_its_failed_check_and_always_shuts_down(
        tmp_path, mode, status, attempts, reports):
    source = (ROOT / ".github/scripts/engine-e2e-runner.sh").read_text()
    functions = "\n".join(
        re.findall(r"^[a-z_]+\(\) \{\n.*?^\}", source,
                   re.MULTILINE | re.DOTALL))
    trap = next(line for line in source.splitlines()
                if line.startswith("trap ") and line.endswith(" EXIT"))
    body = "boot_check=GPU\nfalse\n"
    preflight = mode == "failure" or mode.startswith("report-")
    if not preflight:
        body = source.split("and local SSD'\n",
                            1)[1].replace("cd /opt/actions-runner",
                                          'cd "$TEST_DIR"')
    clock = ('case $TEST_MODE in *-deadline) echo 1999999999;; '
             'report-expired) echo 2000000000;; *) echo 1000000000;; esac')
    commands = dict(
        date=clock, mktemp='exec /usr/bin/mktemp "$TEST_DIR/token.XXXXXX"',
        shutdown='printf shutdown >> "$TEST_DIR/shutdown.log"', chown="exit 0",
        runuser='[[ $TEST_MODE != configuration && '
        '( $TEST_MODE != job-exit || " $* " != *" ./run.sh "* ) ]] || exit 7',
        sleep='(( $1 == 0 )) || printf "%s\\n" "$1" >> "$TEST_DIR/sleeps"')
    for name, command in commands.items():
        (tmp_path / name).write_text("#!/bin/bash\n" + command + "\n")
        (tmp_path / name).chmod(0o755)
    (tmp_path / "aws").write_text("""#!/usr/bin/python3
import json, os, sys
from pathlib import Path
payload = json.loads(sys.argv[sys.argv.index('--payload') + 1])
with (Path(os.environ['TEST_DIR']) / 'events').open('a') as output:
    output.write(json.dumps(payload) + '\\n')
mode = os.environ['TEST_MODE']
attempt = sum(json.loads(line)['kind'] == payload['kind'] for line in
    (Path(os.environ['TEST_DIR']) / 'events').read_text().splitlines())
registration = payload['kind'] == 'registration'
if (registration and (mode in ('registration-transport', 'registration-deadline') or
        (mode == 'transport-recovery' and attempt < 3) or (mode == 'drained-recovery' and attempt < 2))) or (
        not registration and (mode in ('report-failure', 'report-deadline') or
        (mode == 'report-transport-recovery' and attempt < 2))):
    raise SystemExit(31)
Path(sys.argv[-1]).write_text(json.dumps(None if not registration or mode in ('drained', 'drained-recovery') else 'fixture-token'))
failed = (registration and (mode == 'registration-function' or (mode == 'function-recovery' and attempt < 3))) or (
    not registration and mode == 'report-function-recovery' and attempt < 3)
print('Unhandled' if failed else 'None')
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
        POOL_ATTEMPT="2", SOURCE_SHA="a" * 40,
        CAPACITY_DEADLINE=("invalid" if mode == "report-invalid-deadline" else
                           "2000000000"), INSTANCE_TYPE="g7e.8xlarge",
        LABEL=cases[0]["label"], POOL_CASES=json.dumps(cases),
        CAPACITY_FUNCTION="fixture", AWS_REGION="us-west-2",
        actual_instance="g7e.8xlarge", TIER="weekly"), capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == status, result.stdout + result.stderr
    lines = (tmp_path / "events").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert [event["kind"] for event in events
            ] == ["registration"] * attempts + ["bootstrap"] * reports
    sleeps = tmp_path / "sleeps"
    expected_sleeps = ["5", "10"][:max(0, attempts - 1)]
    expected_sleeps.extend(["5", "10"][:max(0, reports - 1)])
    assert (sleeps.read_text().splitlines()
            if sleeps.exists() else []) == expected_sleeps
    if events:
        assert events[0]["cases"] == cases
    if reports:
        check = ("GPU" if preflight else "runner configuration"
                 if mode == "configuration" else "runner registration")
        error = f"Bootstrap failed: {check} (exit {status})"
        assert all(event["error"] == error for event in events[attempts:])
        assert error in result.stderr
    if mode in ("report-failure", "report-deadline", "report-expired",
                "report-invalid-deadline"):
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
