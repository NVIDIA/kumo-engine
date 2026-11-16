"""Process isolation and failure propagation without Kumo/native dependencies."""

import json
import os
import pwd
import subprocess
import sys

import pytest

from test.kumo_pytorch.benchmarking.campaign import run_campaign


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
    monkeypatch.setenv("POLARS_MAX_THREADS", "4")
    environment = worker_environment(os.getuid(), tmp_path)
    # Check what a real child (and its cache libraries) sees, not only the mapping.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os; from pathlib import Path; "
                "print(json.dumps([str(Path.home()), os.environ['USER'], "
                "os.environ['LOGNAME'], os.path.expanduser('~/.cupy'), "
                "os.environ['POLARS_MAX_THREADS'], os.environ['KUMO_BENCHMARK_CGROUP']]))"
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
        user.pw_dir + "/.cupy",
        "4",
        str(tmp_path),
    ]
    assert os.environ["HOME"] == "/root"  # Supervisor identity is unchanged.
