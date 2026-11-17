"""Exact or explicitly sampled stage peaks for a dedicated Linux cgroup v2.

See https://docs.kernel.org/admin-guide/cgroup-v2.html (memory.peak).
Kernel reset and read use the same descriptor; sampled mode reads memory.current.
No groups, limits or memberships are
changed here. The supplied group must contain only this workload and children.
"""

import itertools
import math
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


def _stage(operation):
    if operation in {"iterator.startup", "batch.request"}:
        return "batch_production"
    if operation in {"ingestion", "materialization"}:
        return operation
    return None


def _is_child(pid, root):
    visited = set()
    while pid > 0 and pid not in visited:
        if pid == root:
            return True
        visited.add(pid)
        content = Path(f"/proc/{pid}/status").read_text()
        pid = int(
            next(
                line.split()[1]
                for line in content.splitlines()
                if line.startswith("PPid:")
            )
        )
    return False


class CgroupMemory:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def validate(self):
        if sys.platform != "linux":
            raise RuntimeError("Stage cgroup peaks require Linux cgroup v2")
        if not (self.path / "cgroup.controllers").is_file():
            raise ValueError("Not a cgroup v2 directory")
        groups = [self.path / "cgroup.procs", *self.path.rglob("*/cgroup.procs")]
        pids = {int(pid) for file in groups for pid in file.read_text().split()}
        current = os.getpid()
        if current not in pids:
            raise ValueError("Benchmark process is outside the selected memory cgroup")
        if not all(_is_child(pid, current) for pid in pids):
            raise ValueError(
                "Memory cgroup contains processes outside the benchmark workload"
            )

    @contextmanager
    def stage(self, operation, recorder):
        stage = _stage(operation)
        if stage is None:
            yield
            return
        metric = stage + ".peak_host_bytes"
        descriptor = None
        failure = None
        started = recorder.clock()
        try:
            self.validate()
            descriptor = os.open(self.path / "memory.peak", os.O_RDWR)
            os.write(descriptor, b"0")
        except (OSError, ValueError, RuntimeError) as error:
            failure = f"{type(error).__name__}: {error}"
        completed = False
        try:
            yield
            completed = True
        finally:
            value = None
            try:
                if failure is None and descriptor is not None:
                    self.validate()
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    value = int(os.read(descriptor, 128).strip())
                    if value < 0:
                        raise ValueError("Negative memory.peak reading")
            except (OSError, ValueError, RuntimeError) as error:
                failure = f"{type(error).__name__}: {error}"
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            details = {
                "method": "cgroup_v2_memory_peak_same_fd_reset",
                "memory_scope": "cgroup_and_descendants_including_file_cache_and_kernel_charges",
                "operation": operation,
                "cgroup": str(self.path),
                "reset_start_ns": started,
                "read_end_ns": recorder.clock(),
                "completed_operation": completed,
                "includes_live_allocations_and_observation_overhead": True,
            }
            if not completed:
                recorder.measure(
                    metric,
                    status="error",
                    reason="Measured operation failed",
                    **details,
                )
            elif failure:
                recorder.measure(
                    metric, status="unavailable", reason=failure, **details
                )
            else:
                recorder.measure(metric, value, **details)


class SampledCgroupMemory(CgroupMemory):
    """Explicit sampled maxima, including native/descendant cgroup charges.

    Reads memory.current only; never resets or substitutes lifetime peaks.
    Boundary reads and periodic reads are retained, including sampling gaps.
    """

    metric_suffix = ".peak_host_bytes"
    method = "sampled_cgroup_v2_memory_current"
    scope = "cgroup_and_descendants_including_file_cache_and_kernel_charges"

    def __init__(self, path, *, interval=0.01, max_samples=100000):
        super().__init__(path)
        if not math.isfinite(interval) or interval <= 0 or max_samples < 2:
            raise ValueError(
                "Positive memory interval and at least two samples required"
            )
        self.interval = interval
        self.max_samples = max_samples

    def _read_current(self):
        value = int((self.path / "memory.current").read_text())
        if value < 0:
            raise ValueError("Negative memory.current reading")
        return value

    @contextmanager
    def stage(self, operation, recorder):
        stage = _stage(operation)
        if stage is None:
            yield
            return
        samples, failures = [], []
        stop = threading.Event()
        thread = None
        boundaries = {}

        def sample(kind):
            try:
                if len(samples) >= self.max_samples:
                    raise RuntimeError("Host-memory sample capacity exceeded")
                start = recorder.clock()
                value = self._read_current()
                samples.append(
                    {
                        "start_ns": start,
                        "end_ns": recorder.clock(),
                        "value": value,
                        "kind": kind,
                    }
                )
                return True
            except (OSError, ValueError, RuntimeError) as error:
                failures.append(f"{type(error).__name__}: {error}")
                return False

        def poll():
            while not stop.wait(self.interval):
                if not sample("periodic"):
                    return

        try:
            self.validate()
            if sample("start_boundary"):
                thread = threading.Thread(target=poll, daemon=True)
                thread.start()
        except (OSError, ValueError, RuntimeError) as error:
            failures.append(f"{type(error).__name__}: {error}")
        completed = False
        try:
            yield boundaries
            completed = True
        finally:
            stop.set()
            if thread is not None and thread.ident is not None:
                thread.join()
            if not failures:
                sample("end_boundary")
                try:
                    self.validate()
                except (OSError, ValueError, RuntimeError) as error:
                    failures.append(f"{type(error).__name__}: {error}")
            details = {
                "method": self.method,
                "precision": "sampled_maximum_not_kernel_high_water_mark",
                "limitations": "Allocations between samples may be missed; boundary reads include observation overhead",
                "memory_scope": self.scope,
                "operation": operation,
                "cgroup": str(self.path),
                "sample_interval_seconds": self.interval,
                "sample_count": len(samples),
                "periodic_sample_count": sum(s["kind"] == "periodic" for s in samples),
                "max_gap_ns": max(
                    (
                        b["start_ns"] - a["end_ns"]
                        for a, b in itertools.pairwise(samples)
                    ),
                    default=0,
                ),
                "samples": samples,
                "operation_boundaries": boundaries,
                "completed_operation": completed,
                "includes_live_allocations_and_observation_overhead": True,
            }
            if not completed:
                recorder.measure(
                    stage + self.metric_suffix,
                    status="error",
                    reason="Measured operation failed",
                    **details,
                )
            elif failures:
                recorder.measure(
                    stage + self.metric_suffix,
                    status="unavailable",
                    reason="; ".join(failures),
                    **details,
                )
            else:
                recorder.measure(
                    stage + self.metric_suffix,
                    max(s["value"] for s in samples),
                    **details,
                )


class SampledAnonymousMemory(SampledCgroupMemory):
    """Sample memory.stat anon, including descendants; exclude file/shmem charges.

    This is a sampled peak, independent of the total host peak's timestamp.
    Kernel statistics can lag allocations; it is not an exact allocation peak.
    """

    metric_suffix = ".peak_anonymous_bytes"
    method = "sampled_cgroup_v2_memory_stat_anon"
    scope = "cgroup_and_descendants_anonymous_memory_excluding_file_and_shmem"

    def _read_current(self):
        fields = dict(
            line.split()
            for line in (self.path / "memory.stat").read_text().splitlines()
        )
        if "anon" not in fields:
            raise ValueError("memory.stat has no anon counter")
        value = int(fields["anon"])
        if value < 0:
            raise ValueError("Negative memory.stat anon reading")
        return value
