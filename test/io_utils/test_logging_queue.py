# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging as py_logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from io_utils import logging

LOGGER_NAME = 'test_logging_queue'
_FORK_HELPER = Path(__file__).with_name('logging_queue_fork_helper.py')
_MULTIPROCESSING_HELPER = Path(__file__).with_name(
    'multiprocessing_log_helper.py')
_REPO_ROOT = Path(__file__).parents[2]
_FORK_CHILD_TIMEOUT_S = 3.0
_FORK_HELPER_TIMEOUT_S = 10.0


def _read(path, expected, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = path.read_text()
        if text.count('\n') >= expected:
            return text
        time.sleep(0.01)
    return path.read_text()


def test_multiprocessing_file_handler_flushes_records_at_exit(tmp_path):
    path = tmp_path / 'multiprocessing.log'
    result = subprocess.run(
        [sys.executable,
         str(_MULTIPROCESSING_HELPER),
         str(path)],
        capture_output=True,
        text=True,
        timeout=15.0,
        env={**os.environ, 'PYTHONPATH': str(_REPO_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    assert path.read_text().splitlines() == [
        'from_parent',
        'from_child',
        *(f'burst_{index}' for index in range(2000)),
    ]


def test_multiprocessing_file_handler_ignores_records_after_close(
        tmp_path, capsys):
    path = tmp_path / 'closed.log'
    handler = logging.MultiProcessingLog(path, 'a', 1048576, 1)
    handler.close()

    handler.emit(
        py_logging.LogRecord(LOGGER_NAME, py_logging.INFO, __file__, 1,
                             'after close', (), None))

    assert capsys.readouterr().err == ''
    assert path.read_text() == ''


@pytest.fixture
def routed(tmp_path):
    path = tmp_path / 'kumo.log'
    handler = py_logging.FileHandler(path)
    handler.setFormatter(py_logging.Formatter('%(message)s'))
    logger = py_logging.getLogger(LOGGER_NAME)
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False
    logging._route_handlers_through_queue([LOGGER_NAME])
    yield logger, path
    logging._stop_queue_listener()
    logging._ROUTED_HANDLERS.clear()
    handler.close()
    logger.handlers = []


def _terminate_helper_group(process):
    try:
        os.killpg(process.pid, signal.SIGUSR1)
    except ProcessLookupError:
        # The helper group exited between timeout detection and signaling.
        pass
    try:
        return process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The helper group exited during the stack-dump grace period.
        pass
    try:
        return process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The helper group exited during the termination grace period.
        pass
    try:
        return process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        return '', 'helper process group did not exit after SIGKILL\n'


def _run_fork_helper(tmp_path, *, child_timeout=_FORK_CHILD_TIMEOUT_S,
                     simulate_stuck=False, force_quiesce_timeout=False):
    log_path = tmp_path / 'fork-helper.log'
    report_path = tmp_path / 'fork-helper-report.json'
    command = [
        sys.executable,
        str(_FORK_HELPER),
        '--log-path',
        str(log_path),
        '--report-path',
        str(report_path),
        '--timeout',
        str(child_timeout),
    ]
    if simulate_stuck:
        command.append('--simulate-stuck')
    if force_quiesce_timeout:
        command.append('--force-quiesce-timeout')

    started_at = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=_FORK_HELPER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_helper_group(process)
        report = {
            'status': 'helper_timed_out',
            'helper_pid': process.pid,
            'helper_exitcode': process.returncode,
        }
    else:
        if report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            report = {
                'status': 'missing_report',
                'helper_pid': process.pid,
                'helper_exitcode': process.returncode,
            }
    report['outer_elapsed_seconds'] = time.monotonic() - started_at
    report['helper_exitcode'] = process.returncode
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report, stdout, stderr, report_path


def _preserve_fork_diagnostics(nodeid, report, stdout, stderr, report_path):
    from test import _flake_diagnostics

    out = _flake_diagnostics._artifacts_dir_for(nodeid)
    if report_path.exists():
        shutil.copyfile(report_path, out / 'fork-helper-report.json')
    else:
        (out / 'fork-helper-report.json').write_text(
            json.dumps(report, indent=2, sort_keys=True))
    (out / 'fork-helper-stdout.txt').write_text(stdout)
    (out / 'fork-helper-stderr.txt').write_text(stderr)


def _assert_fork_helper_completed(request, report, stdout, stderr, report_path):
    events = [event.get('event') for event in report.get('events', [])]
    parent_before_fork = next((event for event in report.get('events', [])
                               if event.get('event') == 'parent_before_fork'),
                              {})
    threads = parent_before_fork.get('threads', [])
    failures = []
    if report.get('status') != 'completed':
        failures.append(f"status={report.get('status')!r}")
    if report.get('child_exitcode') != 0:
        failures.append(f"child_exitcode={report.get('child_exitcode')!r}")
    if report.get('child_alive_after_cleanup'):
        failures.append('child was alive after cleanup')
    if parent_before_fork.get('requested_start_method') != 'fork':
        failures.append('helper did not use the fork start method')
    if len(threads) != 2:
        failures.append(f'helper forked with unexpected threads: {threads!r}')
    for expected in (
            'child_at_fork_callbacks_started',
            'child_at_fork_callbacks_completed',
            'child_target_started',
            'child_before_log',
            'child_after_log',
            'child_after_second_log',
            'child_after_logging_shutdown',
            'child_target_completed',
    ):
        if expected not in events:
            failures.append(f'missing checkpoint {expected!r}')
    text = report.get('log_text', '')
    if 'from_parent' not in text:
        failures.append('parent record was not written')
    if 'from_child' not in text:
        failures.append('child record was not written')

    if failures:
        _preserve_fork_diagnostics(request.node.nodeid, report, stdout, stderr,
                                   report_path)
        pytest.fail(
            '; '.join(failures) + '\n' +
            json.dumps(report, indent=2, sort_keys=True), pytrace=False)


def test_fork_child_records_are_written(tmp_path, request):
    report, stdout, stderr, report_path = _run_fork_helper(tmp_path)
    _assert_fork_helper_completed(request, report, stdout, stderr, report_path)


def test_fork_helper_bounds_and_diagnoses_a_stuck_child(tmp_path):
    report, _, stderr, _ = _run_fork_helper(tmp_path, child_timeout=1.5,
                                            simulate_stuck=True)
    events = [event.get('event') for event in report['events']]
    assert report['status'] == 'timed_out'
    assert report['outer_elapsed_seconds'] < _FORK_HELPER_TIMEOUT_S
    assert not report['child_alive_after_cleanup']
    assert 'child_simulated_stuck' in events
    assert 'parent_requested_child_stack' in events
    assert 'parent_terminated_child' in events
    assert '_child_logs' in stderr


def test_fork_quiesce_timeout_disables_unsafe_child_routes(tmp_path):
    report, _, stderr, _ = _run_fork_helper(tmp_path,
                                            force_quiesce_timeout=True)
    events = [event.get('event') for event in report['events']]
    assert report['status'] == 'completed'
    assert report['outer_elapsed_seconds'] < _FORK_HELPER_TIMEOUT_S
    assert report['child_exitcode'] == 0
    assert not report['child_alive_after_cleanup']
    assert 'child_after_log' in events
    assert 'child_after_second_log' in events
    assert 'child_after_logging_shutdown' in events
    assert 'stderr_saturated' in events
    assert 'parent_after_fork' in events
    assert 'Exception ignored in atfork callback' not in stderr
    assert 'from_child' not in report['log_text']


def test_fork_child_restores_routes_after_listener_was_stopped(routed):
    logger, path = routed
    logging._stop_queue_listener()
    assert logging._QUEUE_LISTENER is None
    assert logging._ROUTED_HANDLERS

    # Model the successful pre-fork callback; this test exercises the child
    # routing decision without actually cloning the pytest process.
    logging._AT_FORK_STATE.handlers_quiesced = True
    logging._AT_FORK_STATE.failure_reason = None
    logging._restore_direct_handlers_after_fork()

    logger.info('after_stopped_listener')
    assert 'after_stopped_listener' in _read(path, 1)


def test_concurrent_fork_callbacks_keep_lock_ownership_isolated(
        routed, monkeypatch):
    logger, path = routed
    handler, = logging._ROUTED_HANDLERS[LOGGER_NAME]
    handler_lock = handler.lock
    assert handler_lock is not None
    monkeypatch.setattr(logging, '_FORK_HANDLER_LOCK_TIMEOUT_S', 0.05)
    first_prepared = threading.Event()
    release_first = threading.Event()
    second_completed = threading.Event()
    errors = []

    def first_callback_sequence():
        prepared = False
        try:
            logging._acquire_routed_handler_locks_before_fork()
            prepared = True
            first_prepared.set()
            if not release_first.wait(5.0):
                raise RuntimeError('first callback was not released')
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(exc)
        finally:
            if prepared:
                logging._release_routed_handler_locks_after_fork_parent()

    def second_callback_sequence():
        prepared = False
        try:
            if not first_prepared.wait(5.0):
                raise RuntimeError('first callback did not acquire its locks')
            logging._acquire_routed_handler_locks_before_fork()
            prepared = True
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(exc)
        finally:
            if prepared:
                logging._release_routed_handler_locks_after_fork_parent()
            second_completed.set()

    first = threading.Thread(target=first_callback_sequence)
    second = threading.Thread(target=second_callback_sequence)
    first.start()
    second.start()
    routing_lock_acquired = False
    handler_lock_acquired = False
    try:
        assert first_prepared.wait(5.0)
        assert second_completed.wait(5.0)
        routing_lock_acquired = logging._QUEUE_LISTENER_LOCK.acquire(
            timeout=0.05)
        if routing_lock_acquired:
            logging._QUEUE_LISTENER_LOCK.release()
        handler_lock_acquired = handler_lock.acquire(timeout=0.05)
        if handler_lock_acquired:
            handler_lock.release()
    finally:
        release_first.set()
        first.join(5.0)
        second.join(5.0)

    assert not routing_lock_acquired
    assert not handler_lock_acquired
    assert not errors
    assert not first.is_alive()
    assert not second.is_alive()
    logger.info('after_failed_quiesce')
    assert 'after_failed_quiesce' in _read(path, 1)


def test_deferred_args_are_resolved_at_call_time(routed):
    logger, path = routed
    payload = {'stage': 'before'}
    logger.info('stage=%s', payload)
    payload['stage'] = 'after'
    assert "{'stage': 'before'}" in _read(path, 1)


def test_handle_does_not_take_the_handler_lock(routed):
    logger, _ = routed
    handler, = [
        h for h in logger.handlers if isinstance(h, logging._RecordQueueHandler)
    ]
    handler_lock = handler.lock
    assert handler_lock is not None
    handler_lock.acquire()
    try:
        done = threading.Event()
        threading.Thread(
            target=lambda: (logger.info('unlocked'), done.set())).start()
        assert done.wait(5.0)
    finally:
        handler_lock.release()


def test_queue_is_bounded_and_drops_are_reported(routed, monkeypatch):
    logger, path = routed
    logging._stop_queue_listener()
    monkeypatch.setattr(logging, '_QUEUE_MAX_RECORDS', 4)
    monkeypatch.setattr(logging, '_DROPPED_RECORDS', 0)
    monkeypatch.setattr(logging, '_REPORTED_DROPS', 0)
    for i in range(50):
        logger.info('record %d', i)
    assert logging._DROPPED_RECORDS == 46

    logging._restore_direct_handlers()
    logging._route_handlers_through_queue([LOGGER_NAME])
    logger.info('after')
    text = _read(path, 2)
    assert 'logging queue full: dropped 46 records' in text


def test_listener_survives_a_raising_handler(routed):
    logger, path = routed
    listener = logging._QUEUE_LISTENER
    assert listener is not None
    original = tuple(listener.handlers)

    class _Boom(py_logging.Handler):
        def handle(self, record):
            raise RuntimeError('boom')

    listener.handlers = (_Boom(), ) + original
    logger.info('first')
    # The handler after the raising one must still get this record.
    assert 'first' in _read(path, 1)

    listener.handlers = original
    logger.info('second')
    assert 'second' in _read(path, 2)
    assert listener._thread is not None
    assert listener._thread.is_alive()


def test_stop_bounds_the_join_on_a_wedged_sink(routed, monkeypatch):
    logger, _ = routed
    monkeypatch.setattr(logging, '_LISTENER_JOIN_TIMEOUT_S', 0.5)
    blocked = threading.Event()

    class _Wedged(py_logging.Handler):
        def handle(self, record):
            blocked.set()
            time.sleep(30)

    listener = logging._QUEUE_LISTENER
    assert listener is not None
    listener.handlers = (_Wedged(), )
    logger.info('wedge')
    assert blocked.wait(5.0)
    start = time.monotonic()
    logging._stop_queue_listener()
    assert time.monotonic() - start < 5.0


def test_drain_logs_is_idempotent(routed):
    logger, path = routed
    logging.drain_logs()
    logging.drain_logs()
    logger.info('once')
    assert path.read_text().count('once') == 1


def test_a_logger_the_next_route_omits_keeps_logging(routed, tmp_path):
    logger, path = routed
    other = py_logging.getLogger(f'{LOGGER_NAME}_other')
    handler = py_logging.FileHandler(tmp_path / 'other.log')
    other.handlers = [handler]
    other.propagate = False
    logging.drain_logs()
    logging._route_handlers_through_queue([other.name])
    try:
        logger.info('direct')
        assert 'direct' in path.read_text()
    finally:
        logging._stop_queue_listener()
        other.handlers = []
        handler.close()


def test_drain_logs_restores_direct_handlers(routed):
    logger, path = routed
    logger.info('queued')
    logging.drain_logs()
    assert not [
        h for h in logger.handlers if isinstance(h, logging._RecordQueueHandler)
    ]
    logger.info('direct')
    text = path.read_text()
    assert 'queued' in text
    assert 'direct' in text
