# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small Linux process-tree supervisor for pooled Kumo CLI activities.

The worker starts this module as a new session leader. The supervisor waits for
an ownership acknowledgement before starting the real command, adopts orphaned
grandchildren, and does not exit until every adopted child has been reaped.
"""
import argparse
import ctypes
import dataclasses
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pool.pidfd import pidfd_open, pidfd_process_exited, pidfd_send_signal

_PR_SET_CHILD_SUBREAPER = 36
_POLL_SECS = 0.05
_termination_signal: int | None = None


@dataclasses.dataclass(frozen=True)
class _SessionMember:
    """One process plus every task state visible in its thread group."""
    pid: int
    leader_state: str
    task_states: tuple[tuple[int, str], ...] | None
    task_group_exited: bool | None = None

    @property
    def has_live_tasks(self) -> bool:
        if self.task_states is None:
            return self.task_group_exited is not True
        if not self.task_states:
            return (self.leader_state not in ('X', 'Z')
                    or self.task_group_exited is not True)
        return any(state not in ('X', 'Z') for _, state in self.task_states)


def _become_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _record_termination(sig: int, _frame) -> None:
    global _termination_signal
    _termination_signal = sig


def _task_states(pid: int) -> tuple[tuple[int, str], ...] | None:
    states = []
    try:
        with os.scandir(f'/proc/{pid}/task') as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    stat = Path(
                        f'/proc/{pid}/task/{entry.name}/stat').read_bytes()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                fields = stat[stat.rfind(b')') + 2:].split()
                if fields:
                    states.append((int(entry.name), fields[0].decode()))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return tuple(states)


def _session_members() -> list[_SessionMember]:
    session_id = os.getsid(0)
    members = []
    with os.scandir('/proc') as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            try:
                stat = Path(f'/proc/{pid}/stat').read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            fields = stat[stat.rfind(b')') + 2:].split()
            if len(fields) >= 4 and int(fields[3]) == session_id:
                leader_state = fields[0].decode()
                task_states = _task_states(pid)
                task_group_exited = None
                if (task_states is None
                        or (not task_states and leader_state in ('X', 'Z'))):
                    try:
                        task_group_exited = pidfd_process_exited(pid)
                    except ProcessLookupError:
                        continue
                    except OSError:
                        pass
                members.append(
                    _SessionMember(
                        pid,
                        leader_state,
                        task_states,
                        task_group_exited,
                    ))
    return members


def _signal_members(sig: int) -> None:
    for member in _session_members():
        if not member.has_live_tasks:
            continue
        try:
            pidfd = pidfd_open(member.pid)
        except ProcessLookupError:
            continue
        try:
            pidfd_send_signal(pidfd, sig)
        except ProcessLookupError:
            # The session member exited after enumeration.
            pass
        finally:
            os.close(pidfd)


def _reap_nonblocking(main_pid: int,
                      main_status: int | None) -> int | None:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return main_status
        if pid == 0:
            return main_status
        if pid == main_pid:
            main_status = status


def _wait_for_members(main_pid: int, main_status: int | None, sig: int,
                      timeout: float) -> tuple[int | None, bool]:
    deadline = time.monotonic() + timeout
    while True:
        main_status = _reap_nonblocking(main_pid, main_status)
        members = _session_members()
        if not members:
            return main_status, True
        _signal_members(sig)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return main_status, False
        time.sleep(min(_POLL_SECS, remaining))


def _wait_for_main(main_pid: int) -> int:
    main_status = None
    while main_status is None:
        try:
            pid, status = os.waitpid(-1, 0)
        except InterruptedError:
            continue
        if pid == main_pid:
            main_status = status
    return main_status


def _status_to_exit_code(status: int) -> int:
    exit_code = os.waitstatus_to_exitcode(status)
    return exit_code if exit_code >= 0 else 128 - exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ready-fd', type=int, required=True)
    parser.add_argument('--ack-fd', type=int, required=True)
    parser.add_argument('--term-grace', type=float, required=True)
    parser.add_argument('--kill-grace', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] != ['--'] or len(args.command) == 1:
        parser.error('expected -- followed by a command')
    args.command = args.command[1:]
    return args


def main() -> int:
    args = _parse_args()
    try:
        _become_subreaper()
    except Exception:
        # The parent has not acknowledged ownership, so the real command has
        # not started. Advertise graceful-degradation instead of failing the
        # activity before its CLI launch.
        os.write(args.ready_fd, b'U')
        os.close(args.ready_fd)
        os.close(args.ack_fd)
        return 126
    signal.signal(signal.SIGTERM, _record_termination)
    signal.signal(signal.SIGINT, _record_termination)

    os.write(args.ready_fd, b'R')
    os.close(args.ready_fd)
    acknowledgement = os.read(args.ack_fd, 1)
    os.close(args.ack_fd)
    if acknowledgement != b'G':
        return 125
    if _termination_signal is not None:
        return 128 + _termination_signal

    child = subprocess.Popen(args.command)
    main_status = _wait_for_main(child.pid)

    # A normally exiting CLI may have left native descendants behind. Reclaim
    # and reap them before exposing its exit code to the activity parent.
    main_status, quiescent = _wait_for_members(
        child.pid, main_status, signal.SIGTERM, args.term_grace)
    if not quiescent:
        main_status, quiescent = _wait_for_members(
            child.pid, main_status, signal.SIGKILL, args.kill_grace)
    if not quiescent:
        return 124

    if _termination_signal is not None:
        return 128 + _termination_signal
    assert main_status is not None
    return _status_to_exit_code(main_status)


if __name__ == '__main__':
    sys.exit(main())
