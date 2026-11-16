"""Stage high-water accounting for a caller-provisioned Linux cgroup v2.

See https://docs.kernel.org/admin-guide/cgroup-v2.html (memory.peak).
Reset and read use the same descriptor. No groups, limits or memberships are
changed here. The supplied group must contain only this workload and children.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path


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
        stage = (
            "batch_production"
            if operation in {"iterator.startup", "batch.request"}
            else operation
            if operation in {"ingestion", "materialization"}
            else None
        )
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
