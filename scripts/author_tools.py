# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small shared receipts and process handling for local author tools."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git_state(cwd: Path) -> dict:
    def git(*args):
        result = subprocess.run(['git', '-C', str(cwd), *args],
                                capture_output=True, timeout=10)
        if result.returncode:
            raise ValueError('Not a readable Git worktree')
        return result.stdout

    root = {}
    try:
        root['root'] = git('rev-parse', '--show-toplevel').decode().strip()
        return {
            **root,
            'revision':
            git('rev-parse', 'HEAD').decode().strip(),
            'dirty':
            bool(git('status', '--porcelain')),
            'tracked_diff_sha256':
            hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
        }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return dict(root, revision=None, dirty=None)


def machine() -> dict:
    result = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
        'runner_python': sys.version,
        'runner_executable': sys.executable
    }
    if hasattr(os, 'sched_getaffinity'):
        result['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    if hasattr(os, 'sysconf'):
        try:
            result['physical_memory_bytes'] = os.sysconf(
                'SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (OSError, ValueError):
            pass
    return result


def new_output(path: Path, excluded: tuple[Path, ...] = ()) -> Path:
    path = path.expanduser().resolve()
    if any(path.is_relative_to(cwd.resolve()) for cwd in excluded):
        raise ValueError(
            'Report directory must be outside the measured worktrees')
    path.mkdir(parents=True, exist_ok=False)
    path.chmod(0o700)
    return path


def group_running(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    proc = Path('/proc')
    if not proc.is_dir():
        return True
    # A killed orphan can remain as an init-owned zombie. It cannot do work.
    for path in proc.glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] not in ['Z', 'X']:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def run_process(argv: list[str], cwd: Path, log: Path, timeout: float,
                env: dict | None = None) -> dict:
    """Time a complete fresh process; stop its process group on every exit path."""
    if os.name != 'posix':
        raise ValueError(
            'These runners require POSIX process groups (use WSL on Windows)')
    process = None
    row = {
        'argv': argv,
        'cwd': str(cwd),
        'log': log.name,
        'timeout_seconds': timeout,
        'status': 'failed'
    }
    started = time.perf_counter()
    try:
        with log.open('xb') as stream:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                row['exit_code'] = process.wait(timeout=timeout)
                row['status'] = 'ok' if row['exit_code'] == 0 else 'failed'
            except subprocess.TimeoutExpired:
                row['status'] = 'timeout'
                row['exit_code'] = None
            row['wall_seconds'] = time.perf_counter() - started
    except OSError as error:
        row['error'] = str(error)
        row['exit_code'] = None
        row['wall_seconds'] = time.perf_counter() - started
    finally:
        if process is not None:
            if process.poll() is not None:
                # multiprocessing resource trackers and ordinary worker shutdown
                # can outlive the parent briefly. Let them drain naturally.
                deadline = time.monotonic() + 1
                while group_running(
                        process.pid) and time.monotonic() < deadline:
                    time.sleep(.02)
            if group_running(process.pid):
                if row['status'] == 'ok':
                    row['status'] = 'failed'
                    row['error'] = 'Command left background processes running'
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
    return row


@contextlib.contextmanager
def handle_termination():
    """Let the CLI's normal interruption cleanup handle SIGTERM as well."""
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Terminated')

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
