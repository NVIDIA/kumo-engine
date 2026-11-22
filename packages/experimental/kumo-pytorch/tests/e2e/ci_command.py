# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a CI command and retain its merged console output. Usage: ci_command LOG COMMAND [ARG ...]"""

import os
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from time import monotonic, sleep


def run(log, command, heartbeat=30):
    log.parent.mkdir(parents=True, exist_ok=True)
    start = monotonic()
    next_heartbeat = start + heartbeat
    process = None
    cancelled = None
    kill_at = None

    def send(sig):
        if process is not None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Darwin reports EPERM for a group left with only the unreaped leader.
                if sys.platform != "darwin" or process.poll() is None:
                    raise

    def cancel(sig, _frame):
        nonlocal cancelled
        cancelled = cancelled or sig

    handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        # The child writes directly to disk, including output just before cancellation.
        with log.open("wb", buffering=0) as output, log.open("rb") as reader:
            try:
                process = subprocess.Popen(
                    command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
                )
            except OSError as error:
                output.write(f"{error}\n".encode())
                code = 127 if isinstance(error, FileNotFoundError) else 126
            else:
                while process.returncode is None:
                    if cancelled is None:
                        with suppress(subprocess.TimeoutExpired):
                            process.wait(timeout=min(1, heartbeat))
                    elif kill_at is None:
                        send(cancelled)
                        kill_at = monotonic() + 3
                    elif monotonic() >= kill_at:
                        # Hold the leader unreaped during grace so its group ID cannot be reused.
                        send(signal.SIGKILL)
                        process.wait()
                    else:
                        sleep(0.1)
                    shutil.copyfileobj(reader, sys.stdout.buffer)
                    sys.stdout.buffer.flush()
                    now = monotonic()
                    if process.returncode is None and now >= next_heartbeat:
                        print(f"Still running {command[0]} ({now - start:.1f}s)", flush=True)
                        next_heartbeat = now + heartbeat
                code = process.returncode
            shutil.copyfileobj(reader, sys.stdout.buffer)
    finally:
        if process is not None and process.returncode is None:
            send(signal.SIGKILL)
            process.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if cancelled:
        code = -cancelled
    if code < 0:
        code = 128 - code
    print(f"{command[0]} exited {code} after {monotonic() - start:.1f}s", flush=True)
    return code


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(run(Path(sys.argv[1]), sys.argv[2:]))
