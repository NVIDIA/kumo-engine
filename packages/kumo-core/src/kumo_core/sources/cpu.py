# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

MAX_DOWNLOAD_WORKERS = 32
DOWNLOAD_OVERSUBSCRIPTION = 4

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_CGROUP = Path("/proc/self/cgroup")
_MEMINFO = Path("/proc/meminfo")


def cpu_budget() -> int:
    """``os.cpu_count`` reports the machine, so a pool sized from it
    oversubscribes by the whole ratio of host cores to quota -- 96 to 2 on a
    cgroup-limited worker."""
    try:
        quota = _cgroup_v2_cpus() or _cgroup_v1_cpus()
    except Exception:
        quota = None
    if quota is not None:
        return quota
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def download_workers(worker_ceiling: int | None = None) -> int:
    """Transfers are network-bound and mostly waiting, so oversubscribe;
    capped because past roughly 32 the warehouse became the limit."""
    budget = cpu_budget() * DOWNLOAD_OVERSUBSCRIPTION
    return _capped(min(MAX_DOWNLOAD_WORKERS, budget), worker_ceiling)


def encode_workers(worker_ceiling: int | None = None) -> int:
    """Parquet encode is CPU-bound: more than the budget only adds switches."""
    return _capped(cpu_budget(), worker_ceiling)


def memory_ceiling(per_worker: int,
                   worker_ceiling: int | None = None) -> int | None:
    """As many workers as fit in half the memory free now, or
    ``worker_ceiling`` when that cannot be read."""
    try:
        available = _tightest([_meminfo_available(), *_cgroup_memory()])
    except Exception:
        available = None
    if available is None:
        return worker_ceiling
    return _capped(available // 2 // per_worker, worker_ceiling)


def _meminfo_available() -> int | None:
    for line in (_read(_MEMINFO) or "").splitlines():
        name, _, value = line.partition(":")
        if name == "MemAvailable" and value.split()[0].isdigit():
            return int(value.split()[0]) * 1024
    return None


def _cgroup_memory() -> Iterator[int]:
    """Each limit less the anonymous memory under it: page cache is
    reclaimable, and a download-first read fills it."""
    for mount, relative, limit, used in (
        (_CGROUP_ROOT, _self_cgroup(""), "memory.max", "anon"),
        (_CGROUP_ROOT / "memory", _self_cgroup("memory"),
         "memory.limit_in_bytes", "total_rss"),
    ):
        for directory in _ancestors(mount, relative):
            cap = _read(directory / limit) or ""
            if cap.isdigit():
                yield int(cap) - _stat(directory / "memory.stat", used)


def _stat(path: Path, key: str) -> int:
    for line in (_read(path) or "").splitlines():
        name, _, value = line.partition(" ")
        if name == key and value.isdigit():
            return int(value)
    return 0


def _capped(workers: int, worker_ceiling: int | None) -> int:
    """A ceiling is never a target: a caller dividing one machine between
    several reads knows how many may run, not how big it is."""
    if worker_ceiling is not None:
        workers = min(workers, worker_ceiling)
    return max(1, workers)


def _cgroup_v2_cpus() -> int | None:
    found = []
    for directory in _ancestors(_CGROUP_ROOT, _self_cgroup("")):
        fields = (_read(directory / "cpu.max") or "").split()
        if len(fields) == 2:
            found.append(_cpus(fields[0], fields[1]))
    return _tightest(found)


def _cgroup_v1_cpus() -> int | None:
    relative = _self_cgroup("cpu")
    found = []
    for mount in (_CGROUP_ROOT / "cpu", _CGROUP_ROOT / "cpu,cpuacct"):
        for directory in _ancestors(mount, relative):
            found.append(
                _cpus(
                    _read(directory / "cpu.cfs_quota_us") or "",
                    _read(directory / "cpu.cfs_period_us") or ""))
    return _tightest(found)


def _self_cgroup(controller: str) -> str:
    for line in (_read(_PROC_CGROUP) or "").splitlines():
        _, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if controller in controllers.split(","):
            return path
    return "/"


def _ancestors(mount: Path, relative: str) -> Iterator[Path]:
    """Walk to the root, because the quota may be set on any parent."""
    directory = mount / relative.strip("/")
    while directory != mount and mount in directory.parents:
        yield directory
        directory = directory.parent
    yield mount


def _cpus(quota: str, period: str) -> int | None:
    if not quota.lstrip("-").isdigit() or not period.isdigit():
        return None
    allowance, interval = int(quota), int(period)
    if allowance <= 0 or interval <= 0:
        return None
    return max(1, math.ceil(allowance / interval))


def _tightest(limits: Iterable[int | None]) -> int | None:
    return min((limit for limit in limits if limit is not None), default=None)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None
