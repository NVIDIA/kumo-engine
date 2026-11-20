# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process isolation and failure propagation without Kumo/native dependencies."""

import json
import os
import pwd
import subprocess
import sys
from pathlib import Path
from test.kumo_pytorch.benchmarking.campaign import run_campaign
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "memory_method,profile",
    (("kernel", "full"), ("sampled", "full"), ("kernel", "timing")),
)
def test_fresh_processes_retain_incomplete_coverage(tmp_path, memory_method, profile):
    result = run_campaign(
        "test.kumo_pytorch.checks.test_benchmark:tabular",
        {},
        tmp_path / "runs",
        repetitions=2,
        epochs=2,
        memory_method=memory_method,
        memory_interval=0.02,
        measurement_profile=profile,
    )
    assert result["status"] == "incomplete"
    assert result["independent_repeated_runs"] == "measured"
    assert (
        "pending" if profile == "full" else "outside timing-comparison scope"
    ) in result["cold_and_warm_caches"]
    assert result["measurement_profile"] == profile
    for row in result["runs"]:
        directory = tmp_path / "runs" / row["name"]
        config = json.loads((directory / "config.json").read_text())
        assert config["runtime"]["pid"] == row["pid"] != os.getpid()
        assert config["measurement_profile"] == profile
        assert config["memory_method"] == memory_method
        assert config["memory_interval_seconds"] == 0.02
        assert row["exit_code"] == 2
        assert row["execution_completed"]


def test_unimplemented_reuse_does_not_pass(tmp_path):
    result = run_campaign(
        "test.kumo_pytorch.checks.test_benchmark:tabular",
        {},
        tmp_path / "runs",
        repetitions=2,
        compare_fitting=True,
        epochs=1,
    )
    assert all(c["status"] == "incomplete" for c in result["fitting_comparisons"])


def test_complete_workers_do_not_require_unrequested_fitting(monkeypatch, tmp_path):
    def worker(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "results.json").write_text(json.dumps({
            "status": "complete", "execution_status": "completed",
            "metric_coverage": {"complete": 26, "total": 26},
            "experiments": {
                name: {"status": "measured"}
                for name in ("steady_state", "first_and_later_epochs", "cold_and_warm_caches")
            },
        }))
        (output / "config.json").write_text(json.dumps({"runtime": {"pid": 123}}))
        return SimpleNamespace(pid=123, wait=lambda timeout: 0)

    monkeypatch.setattr(subprocess, "Popen", worker)
    result = run_campaign("fixture:complete", {}, tmp_path / "runs", repetitions=2)
    assert result["status"] == "complete"
    assert result["fitting_comparisons"] == []
    assert [row["mode"] for row in result["runs"]] == ["fresh", "fresh"]


@pytest.mark.parametrize("isolate_memory,code,expected", ((True, 124, True), (True, 7, False), (False, 124, False)))
def test_supervisor_timeout_is_reported(monkeypatch, tmp_path, isolate_memory, code, expected):
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *args, **kwargs: SimpleNamespace(pid=123, wait=lambda timeout: code),
    )
    result = run_campaign(
        "fixture:unused", {}, tmp_path / "runs", repetitions=2,
        isolate_memory=isolate_memory,
    )
    assert result["status"] == "failed"
    assert all(row["timed_out"] is expected for row in result["runs"])


@pytest.mark.parametrize("existing", ("symlink", "file"))
def test_privileged_validation_report_does_not_overwrite_worker_files(monkeypatch, tmp_path, existing):
    from test.kumo_pytorch.benchmarking import validate_isolation

    output = tmp_path / "output"
    target = tmp_path / "protected"
    target.write_text("unchanged")
    calls = []

    def execute(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            return 124
        (output / "worker-validation.json").write_text(json.dumps({"uid": 65534, "checks": []}))
        report = output / "isolation-validation.json"
        if existing == "symlink":
            report.symlink_to(target)
        else:
            report.write_text("worker-owned")
        return 0

    monkeypatch.setattr(validate_isolation, "execute", execute)
    with pytest.raises(FileExistsError):
        validate_isolation.main(["--output", str(output), "--parent", str(tmp_path)])
    assert target.read_text() == "unchanged"
    assert (output / "isolation-validation.json").read_text() == (
        "unchanged" if existing == "symlink" else "worker-owned"
    )


@pytest.mark.parametrize("timeout", (0.001, 30))
def test_worker_timeout_or_import_failure_is_a_failure(tmp_path, timeout):
    result = run_campaign(
        "no_such_benchmark_adapter:factory",
        {},
        tmp_path / "runs",
        repetitions=2,
        epochs=1,
        timeout=timeout,
    )
    assert result["status"] == "failed"
    assert all(not row["execution_completed"] for row in result["runs"])
    assert (tmp_path / "runs/report.md").is_file()


def test_isolated_worker_uses_target_identity_after_sudo(monkeypatch, tmp_path):
    from test.kumo_pytorch.benchmarking.cgroup_run import worker_environment

    user = pwd.getpwuid(os.getuid())
    monkeypatch.setenv("HOME", "/root")
    monkeypatch.setenv("USER", "root")
    monkeypatch.setenv("LOGNAME", "root")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    environment = worker_environment(os.getuid(), tmp_path)
    # Check what a real child (and its cache libraries) sees, not only the mapping.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os; from pathlib import Path; "
                "print(json.dumps([str(Path.home()), os.environ['USER'], "
                "os.environ['LOGNAME'], os.path.expanduser('~/.cache'), "
                "os.environ['OMP_NUM_THREADS'], os.environ['KUMO_BENCHMARK_CGROUP']]))"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        user.pw_dir,
        user.pw_name,
        user.pw_name,
        user.pw_dir + "/.cache",
        "4",
        str(tmp_path),
    ]
    assert os.environ["HOME"] == "/root"  # Supervisor identity is unchanged.


def index_fixture(config, workdir):
    from pathlib import Path
    from test.kumo_pytorch.checks.test_benchmark import tabular

    workload = tabular(config, workdir)

    def prepare(recorder):
        action = config["index_action"]
        path = Path(config["index_dir"])
        with recorder.span(f"index.{action}"):
            if action == "build":
                path.mkdir()
                (path / "identity").write_text("original-index")
            else:
                assert (path / "identity").read_text() == "original-index"

    workload.prepare = prepare
    return workload


def test_one_build_two_reopens_use_same_index_in_fresh_workers(tmp_path):
    result = run_campaign(
        "test.kumo_pytorch.checks.test_benchmark_campaign:index_fixture",
        {},
        tmp_path / "lifecycle",
        index_lifecycle=True,
        epochs=1,
    )
    assert result["status"] == "incomplete"  # Fixture does not supply all metrics.
    checks = result["index_lifecycle"]
    assert [r["action"] for r in checks] == ["build", "reopen", "reopen"]
    assert all(r["status"] == "measured" for r in checks)
    assert len({r["index_path"] for r in checks}) == 1
    assert len({r["pid"] for r in result["runs"]}) == 3


def test_lifecycle_rejects_adapter_that_ignores_index_actions(tmp_path):
    result = run_campaign(
        "test.kumo_pytorch.checks.test_benchmark:tabular",
        {},
        tmp_path / "bad",
        index_lifecycle=True,
        epochs=1,
    )
    assert result["status"] == "failed"
    assert all(r["status"] == "failed" for r in result["index_lifecycle"])
