# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pool sized from the host, not the quota, oversubscribes every read."""

import os

import pytest
from kumo_core.sources import cpu

V2 = "0::/kumo/worker\n"
V1 = "9:cpuset:/kumo\n4:cpu,cpuacct:/kumo/worker\n"


@pytest.fixture
def fake_cgroup(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    proc = tmp_path / "proc-self-cgroup"
    monkeypatch.setattr(cpu, "_CGROUP_ROOT", root)
    monkeypatch.setattr(cpu, "_PROC_CGROUP", proc)

    def install(line, files):
        proc.write_text(line)
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    return install


@pytest.mark.parametrize("line,files,expected", [
    (V2, {
        "kumo/worker/cpu.max": "600000 100000"
    }, 6),
    (V2, {
        "kumo/cpu.max": "200000 100000",
        "kumo/worker/cpu.max": "max 100000",
    }, 2),
    (V2, {
        "kumo/worker/cpu.max": "250000 100000"
    }, 3),
    (V2, {
        "kumo/worker/cpu.max": "50000 100000"
    }, 1),
    (V1, {
        "cpu,cpuacct/kumo/worker/cpu.cfs_quota_us": "400000",
        "cpu,cpuacct/kumo/worker/cpu.cfs_period_us": "100000",
    }, 4),
    (V1, {
        "cpu/kumo/cpu.cfs_quota_us": "150000",
        "cpu/kumo/cpu.cfs_period_us": "100000",
    }, 2),
])
def test_the_quota_this_process_runs_under_is_what_is_budgeted(
        fake_cgroup, line, files, expected):
    """A parent may hold the limit, a fraction of a CPU is still one CPU."""
    fake_cgroup(line, files)
    assert cpu.cpu_budget() == expected


def test_no_quota_or_an_unreadable_one_falls_back_to_the_cpus_on_offer(
        fake_cgroup):
    visible = len(os.sched_getaffinity(0))

    fake_cgroup(V2, {"kumo/worker/cpu.max": "max 100000"})
    assert cpu.cpu_budget() == visible

    fake_cgroup("not a cgroup file\n", {"kumo/worker/cpu.max": "nonsense"})
    assert cpu.cpu_budget() == visible

    fake_cgroup(
        V1, {
            "cpu/kumo/worker/cpu.cfs_quota_us": "-1",
            "cpu/kumo/worker/cpu.cfs_period_us": "100000",
        })
    assert cpu.cpu_budget() == visible


def test_transfers_oversubscribe_the_budget_while_encoding_matches_it(
        fake_cgroup):
    fake_cgroup(V2, {"kumo/worker/cpu.max": "400000 100000"})

    assert cpu.encode_workers() == 4
    assert 4 < cpu.download_workers() <= cpu.MAX_DOWNLOAD_WORKERS


def test_a_large_machine_does_not_open_unbounded_transfers(fake_cgroup):
    fake_cgroup(V2, {"kumo/worker/cpu.max": "9600000 100000"})

    assert cpu.download_workers() == cpu.MAX_DOWNLOAD_WORKERS
    assert cpu.encode_workers() == 96


@pytest.mark.parametrize("workers", [cpu.download_workers, cpu.encode_workers])
def test_an_explicit_count_is_a_ceiling_and_never_a_target(
        fake_cgroup, workers):
    fake_cgroup(V2, {"kumo/worker/cpu.max": "400000 100000"})

    assert workers(3) == 3
    assert workers(1000) == workers()
    assert workers(0) == 1


GiB = 1 << 30
WORKER = 256 << 20


@pytest.mark.parametrize("line,files,meminfo,expected", [
    (V2, {
        "kumo/memory.max": str(4 * GiB),
        "kumo/memory.stat": f"anon {GiB}\nfile {2 * GiB}\n",
        "kumo/worker/memory.max": "max",
    }, 64 * GiB, 6),
    ("7:memory:/kumo/worker\n", {
        "memory/kumo/worker/memory.limit_in_bytes": str(2 * GiB),
        "memory/kumo/worker/memory.stat": "cache 0\ntotal_rss 0\n",
    }, 64 * GiB, 4),
    (V2, {"kumo/worker/memory.max": "max"}, GiB, 2),
    (V2, {"kumo/worker/memory.max": str(100 << 20)}, 64 * GiB, 1),
])
def test_workers_are_capped_by_the_memory_free_to_this_process(
        fake_cgroup, tmp_path, monkeypatch, line, files, meminfo, expected):
    """Page cache is reclaimable, so only anonymous memory counts as used."""
    monkeypatch.setattr(cpu, "_MEMINFO", tmp_path / "meminfo")
    (tmp_path / "meminfo").write_text(f"MemTotal: 1 kB\n"
                                      f"MemAvailable: {meminfo // 1024} kB\n")
    fake_cgroup(line, files)

    assert cpu.memory_ceiling(WORKER) == expected
    assert cpu.memory_ceiling(WORKER, 1) == 1


def test_unreadable_memory_leaves_the_ceiling_as_it_was(
        fake_cgroup, tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "_MEMINFO", tmp_path / "missing")
    fake_cgroup(V2, {"kumo/worker/memory.max": "max"})

    assert cpu.memory_ceiling(WORKER) is None
    assert cpu.memory_ceiling(WORKER, 5) == 5
