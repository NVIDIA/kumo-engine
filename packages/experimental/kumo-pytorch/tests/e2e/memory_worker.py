# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one unprivileged pytest worker in a private delegated Linux memory cgroup."""

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def worker_identity():
    uid, gid = os.geteuid(), os.getegid()
    if uid == 0:
        uid, gid = (int(os.environ.get(key, "0")) for key in ("SUDO_UID", "SUDO_GID"))
    if uid <= 0 or gid <= 0:
        raise ValueError("A root supervisor requires the worker's nonroot SUDO_UID and SUDO_GID")
    return uid, gid


def validate_membership(group):
    """The worker must own every live process in the measured subtree."""
    worker = os.getpid()
    if str(worker) not in (group / "cgroup.procs").read_text().split():
        raise RuntimeError(f"Worker {worker} is outside its memory cgroup: {group}")
    for members in group.rglob("cgroup.procs"):
        for member in map(int, members.read_text().split()):
            pid = member
            while pid != worker:
                if not pid:
                    raise RuntimeError(f"Unrelated PID {member} in memory cgroup {group}")
                try:
                    status = Path(f"/proc/{pid}/status").read_text()
                except FileNotFoundError:
                    if Path(f"/proc/{member}").exists():
                        raise RuntimeError(f"Could not verify ancestor of PID {member}") from None
                    break
                pid = int(
                    next(
                        line.split()[1] for line in status.splitlines() if line.startswith("PPid:")
                    )
                )


def run_isolated(command, *, parent, method, env, cwd, timeout=3600):
    if sys.platform != "linux":
        raise RuntimeError("Worker memory measurements require a Linux cgroup v2")
    parent = Path(parent).resolve()
    if "memory" not in (parent / "cgroup.subtree_control").read_text().split():
        raise RuntimeError(f"Memory controller is not delegated at {parent}")
    uid, gid = worker_identity()
    group = parent / f"kumo-loader-{uuid.uuid4().hex}"
    process = None
    waiting = False
    cancelled = None

    def interrupted(signum, _frame):
        nonlocal cancelled
        cancelled = signum
        # During Popen, defer cancellation until the supervisor owns the child handle.
        if waiting:
            raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        group.mkdir()
        try:
            if not (group / "cgroup.kill").is_file():
                raise RuntimeError(f"Worker cgroup cannot clean up descendants: {group}")
            if os.geteuid() == 0:
                import pwd

                user = pwd.getpwuid(uid)
                env = dict(env, HOME=user.pw_dir, USER=user.pw_name, LOGNAME=user.pw_name)
                group.chmod(0o755)
                if method == "kernel":
                    os.chown(group / "memory.peak", uid, gid)
            env = dict(env, KUMO_PYTORCH_MEMORY_CGROUP=str(group), KUMO_PYTORCH_HOST_MEMORY=method)
            # A fresh, single-threaded interpreter joins the group before importing the workload.
            bootstrap = [sys.executable, "-IS", __file__, str(uid), str(gid), *command]
            process = subprocess.Popen(bootstrap, cwd=cwd, env=env)
            try:
                waiting = True
                if cancelled is not None:
                    raise SystemExit(128 + cancelled)
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                waiting = False
            return subprocess.CompletedProcess(command, code)
        finally:
            events = group / "cgroup.events"
            kill = group / "cgroup.kill"
            if process is not None and process.poll() is None:
                kill.write_text("1")
                # The bootstrap can still be outside the group only before dropping privileges.
                if str(process.pid) not in (group / "cgroup.procs").read_text().split():
                    try:
                        process.kill()
                    except PermissionError:
                        pass  # It entered the group and changed UID during the check.
            if "populated 1" in events.read_text():
                kill.write_text("1")
            if process is not None:
                process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while "populated 1" in events.read_text():
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Worker cgroup is still populated: {group}")
                time.sleep(0.01)
            for directory in sorted(group.rglob("*"), reverse=True):
                if directory.is_dir():
                    directory.rmdir()
            group.rmdir()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if cancelled is not None and sys.exc_info()[0] is None:
            raise SystemExit(128 + cancelled)


if __name__ == "__main__":
    (Path(os.environ["KUMO_PYTORCH_MEMORY_CGROUP"]) / "cgroup.procs").write_text(str(os.getpid()))
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(int(sys.argv[2]))
        os.setuid(int(sys.argv[1]))
    os.execvp(sys.argv[3], sys.argv[3:])
