# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import mmap
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from e2e.datasets.hm import HMDataset, generate, sample_anchors
from tests.e2e import runs


@pytest.mark.parametrize(
    "contents,passes",
    [
        ("<testcase classname='tests.model' name='test_cuda'/>", True),
        ("<testcase classname='tests.model' name='test_cuda'><skipped/></testcase>", False),
        ("<testcase classname='tests.other' name='test_cuda'/>", False),
        ("", False),
    ],
)
def test_cuda_campaign_requires_the_selected_junit_cases(tmp_path, contents, passes):
    path = tmp_path / "junit.xml"
    path.write_text(f"<testsuite>{contents}</testsuite>")
    assert (runs._cuda_junit_error(path, ["tests/model.py::test_cuda"]) is None) == passes


def test_aml_campaign_requires_candidate_workers_and_both_epochs(tmp_path):
    path = tmp_path / "timings.json"
    workers = []
    epochs = [{"epoch": i, "complete": True} for i in range(2)]
    for i, action in enumerate(("build", "reopen", "reopen")):
        result = tmp_path / str(i) / "fresh/l01/results.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"fresh": {"epochs": epochs}}))
        worker = dict(fit="fresh", returncode=0, complete=True, index_action=action)
        workers.append(dict(worker, results=str(result.relative_to(tmp_path))))
    report = dict(workload=runs.TIMING_CASES["l01"], device="cpu", workers=workers)
    report.update(revision="candidate", tracked_diff_sha256=None)
    path.write_text(json.dumps(report))
    runs._aml_evidence("l01-index", path, "candidate")
    with pytest.raises(ValueError, match="candidate"):
        runs._aml_evidence("l01-index", path, "another-revision")
    result.write_text(json.dumps({"fresh": {"epochs": epochs[:1]}}))
    with pytest.raises(ValueError, match="epochs"):
        runs._aml_evidence("l01-index", path, "candidate")
    path.write_text(json.dumps({**report, "workers": workers[:-1]}))
    with pytest.raises(ValueError, match="coverage"):
        runs._aml_evidence("l01-index", path, "candidate")


def test_loader_stage_observer_scopes_calls_and_restores_patches():
    from tests.e2e.loader_timing import stage_observer

    original = str.upper
    target = SimpleNamespace(call=original)

    def broken_setup(observer):
        observer.wrap(target, "call", "first")
        observer.wrap(target, "missing", "second")

    with pytest.raises(AttributeError, match="missing"), stage_observer(broken_setup):
        pytest.fail("observer setup should fail")
    assert target.call is original

    with stage_observer() as observer:
        observer.wrap(target, "call", lambda value: value)
        with pytest.raises(RuntimeError, match="already active"), stage_observer():
            pytest.fail("overlapping observers should be rejected")
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(target.call, "other").result() == "OTHER"
        assert target.call("owner") == "OWNER"
    assert target.call is original
    assert set(observer.stages) == {"owner"}
    assert observer.stages["owner"]["calls"] == 1
    assert observer.stages["owner"]["elapsed_ns"] > 0


def test_fused_native_profile_counts_the_interval_once():
    from tests.e2e.loader_timing import _native_totals

    fused = {"operation": "runtime", "fused": True, "sampling_ns": 7, "retrieval_ns": 7}
    assert _native_totals([[fused]]) == {"sample_and_retrieve_ns": 7}
    assert _native_totals([[fused], []]) is None
    assert _native_totals([[fused], [dict(fused, rows_decoded=4)]]) == {
        "sample_and_retrieve_ns": 14
    }


def _check_host_memory():
    from tests.e2e.host_memory import memory_interval

    assert os.geteuid() != 0
    group = Path(os.environ["KUMO_PYTORCH_MEMORY_CGROUP"])
    peaks = {}
    mib = 1024**2
    for name, size in (("large", 64), ("small", 4), ("child", 48)):
        record = {}
        baseline = int((group / "memory.current").read_text())
        with memory_interval(record, name):
            if name == "child":
                program = f"import time; data = b'x' * {size * mib}; time.sleep(0.2)"
                subprocess.run([sys.executable, "-c", program], check=True, timeout=15)
            else:
                data = b"x" * (size * mib)
                time.sleep(0.2)
                del data
        result = peaks[name] = record["host_memory"]
        (interval,) = result["intervals"]
        assert result["complete"] and interval["sample_count"] > 2
        samples = interval["samples"]
        assert result["peak_anonymous_bytes"] == max(s["anonymous_bytes"] for s in samples)
        if os.environ["KUMO_PYTORCH_HOST_MEMORY"] == "sampled":
            assert result["peak_host_bytes"] == max(s["current_bytes"] for s in samples)
        if name != "small":
            assert result["peak_host_bytes"] >= baseline + size * mib * 0.75
    for key in ("peak_host_bytes", "peak_anonymous_bytes"):
        assert peaks["large"][key] > peaks["small"][key] + 32 * mib
        assert peaks["child"][key] > peaks["small"][key] + 24 * mib


@pytest.mark.parametrize("method", ["kernel", "sampled"])
def test_loader_host_memory(method, pytestconfig):
    from tests.e2e.memory_worker import run_isolated

    parent = pytestconfig.getoption("--cgroup-parent", skip=True)
    program = "from tests.e2e.test_runs import _check_host_memory; _check_host_memory()"
    worker = run_isolated(
        [sys.executable, "-c", program],
        parent=parent,
        method=method,
        cwd=runs.REPOSITORY,
        env=os.environ,
    )
    assert worker.returncode == 0


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux file-page residency")
def test_loader_file_page_cache(tmp_path, pytestconfig):
    from tests.e2e.file_cache import IndexPageCache

    if not pytestconfig.getoption("--cold-warm-cache"):
        pytest.skip("select --cold-warm-cache on a disk-backed Linux filesystem")
    path = tmp_path / "pages"
    contents = bytes(range(256)) * (4 * mmap.PAGESIZE // 256)
    path.write_bytes(contents)
    cache = IndexPageCache(tmp_path)
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as reader,
    ):
        assert reader[:] == contents
        for mode, resident in (("warm", 4), ("cold", 0), ("warm", 4)):
            if mode == "cold":
                (tmp_path / "length-cache").write_bytes(b"created after the first epoch")
            evidence = {}
            cache.condition(mode, evidence)
            assert evidence["status"] == "verified"
            assert (evidence["resident_pages"], evidence["pages"]) == (resident, 4)
            if mode == "cold":
                assert evidence["unconditioned_new_files"] == ["length-cache"]
                assert evidence["files"][0]["released_readonly_mappings"] >= 1
        assert reader[:] == contents
    with path.open("r+b") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE):
        with pytest.raises(RuntimeError, match="writable"):
            cache.condition("cold", {})
    path.write_bytes(contents[:1])
    with pytest.raises(RuntimeError, match="changed"):
        cache.condition("warm", {})


def test_memory_worker_timeout_removes_only_its_cgroup(pytestconfig, tmp_path, capfd):
    from tests.e2e.memory_worker import run_isolated, validate_membership

    parent = pytestconfig.getoption("--cgroup-parent", skip=True)
    (tmp_path / "cgroup.procs").write_text(f"{os.getpid()}\n{os.getppid()}\n")
    with pytest.raises(RuntimeError, match="Unrelated PID"):
        validate_membership(tmp_path)
    command = """
import os, subprocess
from pathlib import Path
from tests.e2e.memory_worker import validate_membership
group = Path(os.environ["KUMO_PYTORCH_MEMORY_CGROUP"])
assert os.getuid() > 0
child = subprocess.Popen(["sleep", "60"], start_new_session=True)
validate_membership(group)
print(group, flush=True)
child.wait()
"""
    with subprocess.Popen(["sleep", "60"]) as sibling:
        try:
            result = run_isolated(
                [sys.executable, "-c", command],
                parent=parent,
                method="sampled",
                env=os.environ,
                cwd=runs.REPOSITORY,
                timeout=2,
            )
            assert result.returncode == 124 and sibling.poll() is None
            group = Path(capfd.readouterr().out.strip())
            assert group.parent == parent.resolve() and not group.exists()
        finally:
            sibling.terminate()


@pytest.fixture(scope="module")
def xs(tmp_path_factory: pytest.TempPathFactory) -> HMDataset:
    return generate("XS", 7, tmp_path_factory.mktemp("hm-xs"))


@pytest.fixture
def trained(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The flow commands ``run`` starts, recorded instead of training."""

    commands: list[list[str]] = []

    def train(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        Path(command[command.index("--work") + 1]).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runs.subprocess, "run", train)
    monkeypatch.delenv("KUMO_PYTORCH_E2E_REUSE_ONLY", raising=False)
    return commands


def sampled(xs: HMDataset, root: Path, fractions: Mapping[str, float], seed: int) -> HMDataset:
    return sample_anchors(xs, fractions, seed, root / "sampled")


def test_a_run_is_reused_only_on_the_same_anchor_sample(
    xs: HMDataset, tmp_path: Path, trained: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs.run(sampled(xs, tmp_path, {"train": 0.25}, 0), tmp_path, "gnn", 0)
    assert len(trained) == 1

    runs.run(sampled(xs, tmp_path, {"train": 0.25}, 0), tmp_path, "gnn", 0)
    assert len(trained) == 1, "the identical sample retrained"

    runs.run(sampled(xs, tmp_path, {"train": 0.5}, 0), tmp_path, "gnn", 0)
    assert len(trained) == 2, "another sample fraction reused the old run"

    runs.run(sampled(xs, tmp_path, {"train": 0.5}, 1), tmp_path, "gnn", 0)
    assert len(trained) == 3, "another sample seed reused the old run"

    monkeypatch.setenv("KUMO_PYTORCH_E2E_REUSE_ONLY", "1")
    runs.run(sampled(xs, tmp_path, {"train": 0.5}, 1), tmp_path, "gnn", 0)
    with pytest.raises(RuntimeError, match="no finished run"):
        runs.run(sampled(xs, tmp_path, {"train": 0.25}, 1), tmp_path, "gnn", 0)
    with pytest.raises(RuntimeError, match="no finished run"):
        runs.run(sampled(xs, tmp_path, {"train": 0.5}, 2), tmp_path, "gnn", 0)
    assert len(trained) == 3


def test_a_run_on_the_full_anchors_is_not_reused_for_a_sample(
    xs: HMDataset, tmp_path: Path, trained: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs.run(xs, tmp_path, "pragma", 1)
    runs.run(xs, tmp_path, "pragma", 1)
    assert len(trained) == 1

    monkeypatch.setenv("KUMO_PYTORCH_E2E_REUSE_ONLY", "1")
    with pytest.raises(RuntimeError, match="no finished run"):
        runs.run(sampled(xs, tmp_path, {"train": 0.5}, 0), tmp_path, "pragma", 1)
