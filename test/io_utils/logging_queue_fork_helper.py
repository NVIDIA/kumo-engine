# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated, bounded harness for the logging queue's fork contract."""

import json
import os

_PROGRESS_FD = None


def _checkpoint(event, **details):
    if _PROGRESS_FD is None:
        return
    payload = {
        'event': event,
        'pid': os.getpid(),
        **details,
    }
    try:
        os.write(_PROGRESS_FD,
                 (json.dumps(payload, sort_keys=True) + '\n').encode())
    except OSError:
        # Diagnostics are best-effort if the parent has closed the pipe.
        pass


# This runs before the stdlib logging and Kumo callbacks registered by the
# imports below. A timeout report can therefore distinguish entering the
# at-fork callback sequence from reaching its end.
os.register_at_fork(
    after_in_child=lambda: _checkpoint('child_at_fork_callbacks_started'))

import argparse  # noqa: E402
import contextlib  # noqa: E402
import faulthandler  # noqa: E402
import logging as py_logging  # noqa: E402
import multiprocessing  # noqa: E402
import multiprocessing.popen_fork  # noqa: E402,F401
import multiprocessing.resource_tracker  # noqa: E402,F401
import selectors  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from io_utils import logging  # noqa: E402

LOGGER_NAME = 'test_logging_queue_fork_helper'
_STACK_DUMP_GRACE_S = 0.5
_TERMINATE_GRACE_S = 1.0


class _BlockingStream:
    """A stream whose write remains poisoned in a forked child."""
    def __init__(self, entered, release):
        self._entered = entered
        self._release = release

    def write(self, message):
        self._entered.set()
        self._release.wait()
        return len(message)

    def flush(self):
        pass


def _thread_inventory():
    return [{
        'name': thread.name,
        'ident': thread.ident,
        'daemon': thread.daemon,
        'alive': thread.is_alive(),
    } for thread in threading.enumerate()]


def _child_logs(simulate_stuck):
    _checkpoint('child_target_started')
    _checkpoint('child_before_log')
    py_logging.getLogger(LOGGER_NAME).info('from_child')
    _checkpoint('child_after_log')
    py_logging.getLogger(LOGGER_NAME).info('from_child_again')
    _checkpoint('child_after_second_log')
    py_logging.shutdown()
    _checkpoint('child_after_logging_shutdown')
    if simulate_stuck:
        _checkpoint('child_simulated_stuck')
        threading.Event().wait()
    _checkpoint('child_target_completed')


def _read_progress(progress_fd, buffer, events):
    while True:
        try:
            chunk = os.read(progress_fd, 65536)
        except BlockingIOError:
            break
        if not chunk:
            break
        buffer += chunk
    lines = buffer.split(b'\n')
    for line in lines[:-1]:
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            events.append({
                'event': 'malformed_checkpoint',
                'payload': line.decode(errors='replace'),
                'error': repr(exc),
            })
    return lines[-1]


def _wait_for_child(process, progress_fd, timeout):
    selector = selectors.DefaultSelector()
    selector.register(progress_fd, selectors.EVENT_READ)
    selector.register(process.sentinel, selectors.EVENT_READ)
    events = []
    buffer = b''
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while process.exitcode is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(remaining):
                if key.fileobj == progress_fd:
                    buffer = _read_progress(progress_fd, buffer, events)
                else:
                    process.join(0)
        buffer = _read_progress(progress_fd, buffer, events)
    finally:
        selector.close()

    if timed_out and process.pid is not None:
        events.append({
            'event': 'parent_requested_child_stack',
            'pid': os.getpid(),
            'child_pid': process.pid,
        })
        try:
            os.kill(process.pid, signal.SIGUSR1)
        except ProcessLookupError:
            # The child exited between the liveness check and the signal.
            pass
        process.join(_STACK_DUMP_GRACE_S)

    if process.is_alive():
        events.append({
            'event': 'parent_terminated_child',
            'pid': os.getpid(),
            'child_pid': process.pid,
        })
        process.terminate()
        process.join(_TERMINATE_GRACE_S)
    if process.is_alive():
        events.append({
            'event': 'parent_killed_child',
            'pid': os.getpid(),
            'child_pid': process.pid,
        })
        process.kill()
        process.join(_TERMINATE_GRACE_S)

    buffer = _read_progress(progress_fd, buffer, events)
    if buffer:
        events.append({
            'event': 'trailing_checkpoint_bytes',
            'payload': buffer.decode(errors='replace'),
        })
    return timed_out, events


def _status(timed_out, exitcode, events):
    event_names = {event.get('event') for event in events}
    if timed_out:
        return 'timed_out'
    if exitcode != 0:
        return 'crashed'
    if 'child_target_completed' not in event_names:
        return 'exited_without_completion'
    return 'completed'


def _run(log_path, timeout, simulate_stuck, force_quiesce_timeout):
    global _PROGRESS_FD
    progress_fd, child_progress_fd = os.pipe()
    os.set_blocking(progress_fd, False)
    _PROGRESS_FD = child_progress_fd

    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    blocked = threading.Event()
    release = threading.Event()
    if force_quiesce_timeout:
        Path(log_path).touch()
        handler = py_logging.StreamHandler(_BlockingStream(blocked, release))
        logging._FORK_HANDLER_LOCK_TIMEOUT_S = 0.05
    else:
        handler = py_logging.FileHandler(log_path)
    handler.setFormatter(py_logging.Formatter('%(message)s'))
    logger = py_logging.getLogger(LOGGER_NAME)
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False
    logging._route_handlers_through_queue([LOGGER_NAME])

    saved_stderr = None
    stderr_reader = None
    process = None
    events = []
    started_at = time.monotonic()
    try:
        context = multiprocessing.get_context('fork')
        _checkpoint(
            'parent_before_fork',
            requested_start_method=context.get_start_method(),
            default_start_method=multiprocessing.get_start_method(),
            threads=_thread_inventory(),
        )
        logger.info('from_parent')
        if force_quiesce_timeout and not blocked.wait(2.0):
            raise RuntimeError('listener did not enter the blocking stream')

        if force_quiesce_timeout:
            # No reader drains this pipe until both fork callbacks return.
            # A raw warning write must not undo the bounded lock timeout.
            saved_stderr = os.dup(2)
            stderr_reader, stderr_writer = os.pipe()
            os.set_blocking(stderr_writer, False)
            try:
                while True:
                    os.write(stderr_writer, b'x' * 4096)
            except BlockingIOError:
                pass
            os.set_blocking(stderr_writer, True)
            os.dup2(stderr_writer, 2)
            os.close(stderr_writer)
            _checkpoint('stderr_saturated')

        # Registered last, after multiprocessing's resource-tracker hook and
        # the production logging hook. Reaching it proves those callbacks
        # returned.
        os.register_at_fork(after_in_child=lambda: _checkpoint(
            'child_at_fork_callbacks_completed'))
        process = context.Process(
            target=_child_logs,
            args=(simulate_stuck, ),
        )
        try:
            process.start()
        finally:
            release.set()
        _checkpoint('parent_after_fork', child_pid=process.pid)
        os.close(child_progress_fd)
        child_progress_fd = None
        _PROGRESS_FD = None

        timed_out, events = _wait_for_child(process, progress_fd, timeout)
    finally:
        release.set()
        if process is not None and process.is_alive():
            process.terminate()
            process.join(_TERMINATE_GRACE_S)
        if process is not None and process.is_alive():
            process.kill()
            process.join(_TERMINATE_GRACE_S)
        if saved_stderr is not None:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        if stderr_reader is not None:
            os.close(stderr_reader)
        with contextlib.suppress(OSError):
            os.close(progress_fd)
        if child_progress_fd is not None:
            with contextlib.suppress(OSError):
                os.close(child_progress_fd)
        _PROGRESS_FD = None
        logging.drain_logs()
        handler.close()
        logger.handlers = []
    elapsed = time.monotonic() - started_at

    assert process is not None
    text = Path(log_path).read_text()
    return {
        'status': _status(timed_out, process.exitcode, events),
        'elapsed_seconds': elapsed,
        'helper_pid': os.getpid(),
        'child_pid': process.pid,
        'child_exitcode': process.exitcode,
        'child_alive_after_cleanup': process.is_alive(),
        'events': events,
        'log_text': text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path', required=True)
    parser.add_argument('--report-path', required=True)
    parser.add_argument('--timeout', required=True, type=float)
    parser.add_argument('--simulate-stuck', action='store_true')
    parser.add_argument('--force-quiesce-timeout', action='store_true')
    args = parser.parse_args()

    try:
        report = _run(args.log_path, args.timeout, args.simulate_stuck,
                      args.force_quiesce_timeout)
    except Exception as exc:  # pylint: disable=broad-except
        report = {
            'status': 'helper_error',
            'helper_pid': os.getpid(),
            'error': repr(exc),
            'traceback': traceback.format_exc(),
        }
    Path(args.report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
