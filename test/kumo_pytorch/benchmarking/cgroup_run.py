"""Run one benchmark worker in its own Linux memory cgroup.

The CI supervisor needs root only to provision/migrate the worker. The command
executes as the supplied runner UID/GID. No parent controllers, limits or other
process memberships are changed. The parent must already delegate memory.
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def execute(command, *, parent, uid, gid, timeout):
    if not command or not math.isfinite(timeout) or timeout <= 0 or uid < 0 or gid < 0:
        raise ValueError(
            "A command, finite positive timeout and valid UID/GID are required"
        )
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("Dedicated worker provisioning requires Linux root")
    if threading.active_count() != 1:
        raise RuntimeError("Worker supervisor must have exactly one thread before fork")
    parent = Path(parent).resolve()
    if "memory" not in (parent / "cgroup.subtree_control").read_text().split():
        raise RuntimeError("Parent must already have the memory controller enabled")
    group = parent / f"kumo-benchmark-{uuid.uuid4().hex}"
    group.mkdir()
    process = None
    previous_signal = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        peak = group / "memory.peak"
        if not peak.is_file():
            raise RuntimeError("Worker cgroup has no memory.peak counter")
        os.chown(peak, uid, gid)
        env = dict(os.environ, KUMO_BENCHMARK_CGROUP=str(group))

        def enter():
            (group / "cgroup.procs").write_text(str(os.getpid()))
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)

        process = subprocess.Popen(
            command,
            env=env,
            preexec_fn=enter,  # noqa: PLW1509 -- single-threaded supervisor checked above
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124
    finally:
        # Includes descendants left by a failed adapter; never touches siblings.
        if "populated 1" in (group / "cgroup.events").read_text():
            (group / "cgroup.kill").write_text("1")
        if process is not None:
            process.wait()
        deadline = time.monotonic() + 5
        while "populated 1" in (group / "cgroup.events").read_text():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Worker cgroup still populated: {group}")
            time.sleep(0.01)
        for child in sorted(
            group.rglob("*"), key=lambda path: len(path.parts), reverse=True
        ):
            if child.is_dir():
                child.rmdir()
        group.rmdir()
        signal.signal(signal.SIGTERM, previous_signal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("/sys/fs/cgroup"))
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or args.uid < 0 or args.gid < 0:
        parser.error("A command, positive timeout and valid UID/GID are required")
    try:
        return execute(
            command,
            parent=args.parent,
            uid=args.uid,
            gid=args.gid,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        return 128 + signal.SIGINT


if __name__ == "__main__":
    raise SystemExit(main())
