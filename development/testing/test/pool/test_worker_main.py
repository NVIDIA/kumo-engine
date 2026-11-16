# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``pool.worker_main`` — env-var schema + activity discovery."""
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

import pool.worker_main as worker_main
from pool.activities import (
    SnapshotActivities,
    _run_kumo_subprocess,
    _subprocess_supervisor_capabilities_available,
)
from pool.storage_mode import PoolStorageMode
from pool.task_queues import (
    ACTIVITY_WORKER_POD_NAME_ENV,
    activity_worker_identity,
)
from pool.worker_main import (
    _ACTIVITY_SHUTDOWN_GRACE_SECS,
    _POOL_ACTIVITY_WORKER_POD_NAME,
    _POOL_GPU_CONCURRENCY,
    _POOL_MAX_CONCURRENT_ACTIVITIES,
    _POOL_STORAGE_MODE,
    _POOL_TASK_QUEUE,
    _POOL_WORKER_UID,
    _SIGNAL_SHUTDOWN_TIMEOUT_SECS,
    _discover_activities,
    _finalize_worker,
    _graceful_shutdown_timeout,
    _run_worker_until_shutdown,
)


class _FakeWorker:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.started = asyncio.Event()
        self.complete = asyncio.Event()
        self.shutdown_calls = 0

    async def run(self):
        self.started.set()
        await self.complete.wait()
        if self.failure is not None:
            raise self.failure

    async def shutdown(self):
        self.shutdown_calls += 1
        self.complete.set()


@activity.defn(name='pid1-native-shutdown-activity')
async def _pid1_native_shutdown_activity(_marker: str) -> None:
    await _run_kumo_subprocess(
        'kumo-pred',
        'pid1-real-temporal',
        '/tmp/unused.json',
        {'KUMO_BP_OUTPUT_SPOOL_ROOT': '/tmp/pid1-real-temporal-spool'},
    )


@workflow.defn(name='pid1-native-shutdown-workflow')
class _PID1NativeShutdownWorkflow:
    @workflow.run
    async def run(self, marker: str) -> None:
        await workflow.execute_activity(
            _pid1_native_shutdown_activity,
            marker,
            start_to_close_timeout=timedelta(minutes=2),
        )


_activity_result_done: 'asyncio.Event | None' = None


@activity.defn(name='pid1-activity-result')
async def _pid1_activity_result(marker: str) -> str:
    try:
        return marker
    finally:
        assert _activity_result_done is not None
        _activity_result_done.set()


@workflow.defn(name='pid1-activity-result-workflow')
class _PID1ActivityResultWorkflow:
    @workflow.run
    async def run(self, marker: str) -> str:
        return await workflow.execute_activity(
            _pid1_activity_result,
            marker,
            start_to_close_timeout=timedelta(minutes=1),
        )


def _capture_signal_handlers(monkeypatch):
    loop = asyncio.get_running_loop()
    handlers = {}
    removed = []

    def _add(signum, callback, *args):
        handlers[signum] = (callback, args)

    monkeypatch.setattr(loop, 'add_signal_handler', _add)
    monkeypatch.setattr(loop, 'remove_signal_handler',
                        lambda signum: removed.append(signum) or True)
    return handlers, removed


def test_envvar_declarations_validate_at_import():
    # Reaching this assertion proves that the module-level EnvVar declarations
    # passed pydantic validation (``required`` ⊕ ``default_value``).
    assert _POOL_WORKER_UID.name == 'POOL_WORKER_UID'
    assert _POOL_WORKER_UID.required is True
    assert _POOL_TASK_QUEUE.required is True
    assert _POOL_STORAGE_MODE.default_value == PoolStorageMode.DAEMON_ON_PVC
    assert _POOL_GPU_CONCURRENCY.default_value == 1
    assert _POOL_MAX_CONCURRENT_ACTIVITIES.default_value == 0
    assert worker_main._POOL_NOFILE_SOFT_LIMIT.default_value == 65535
    assert _POOL_ACTIVITY_WORKER_POD_NAME.name == (ACTIVITY_WORKER_POD_NAME_ENV)
    assert _POOL_ACTIVITY_WORKER_POD_NAME.default_value == ''
    assert activity_worker_identity(
        'activity-pod-abc', 'dev-1') == ('pool-worker-activity-pod-abc-dev-1')


def _run_nofile_probe(source: str) -> dict:
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_pool_worker_raises_default_nofile_and_child_inherits():
    result = _run_nofile_probe("""
        import json
        import os
        import resource
        import subprocess
        import sys

        os.environ.pop('POOL_NOFILE_SOFT_LIMIT', None)
        from pool.worker_main import _configure_nofile_soft_limit

        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and hard < 65535:
            raise RuntimeError(f'test host hard nofile limit is only {hard}')
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))

        effective = _configure_nofile_soft_limit()
        child_soft = int(subprocess.check_output([
            sys.executable,
            '-c',
            ('import resource; print(resource.getrlimit('
             'resource.RLIMIT_NOFILE)[0])'),
        ]))
        print(json.dumps({
            'effective': effective,
            'parent_soft': resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            'child_soft': child_soft,
        }))
    """)

    assert result == {
        'effective': 65535,
        'parent_soft': 65535,
        'child_soft': 65535,
    }


def test_pool_worker_nofile_never_lowers_a_higher_soft_limit():
    result = _run_nofile_probe("""
        import json
        import os
        import resource

        os.environ['POOL_NOFILE_SOFT_LIMIT'] = '4096'
        from pool.worker_main import _configure_nofile_soft_limit

        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and hard < 8192:
            raise RuntimeError(f'test host hard nofile limit is only {hard}')
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))

        effective = _configure_nofile_soft_limit()
        print(json.dumps({
            'effective': effective,
            'soft': resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        }))
    """)

    assert result == {'effective': 8192, 'soft': 8192}


def test_pool_worker_nofile_rejects_an_insufficient_hard_limit():
    result = _run_nofile_probe("""
        import json
        import os
        import resource

        os.environ['POOL_NOFILE_SOFT_LIMIT'] = '65535'
        from pool.worker_main import _configure_nofile_soft_limit

        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 4096))
        try:
            _configure_nofile_soft_limit()
        except RuntimeError as exc:
            print(json.dumps({'error': str(exc)}))
        else:
            raise AssertionError('insufficient hard limit was accepted')
    """)

    assert 'hard open-file limit is 4096' in result['error']
    assert 'POOL_NOFILE_SOFT_LIMIT=65535' in result['error']


@pytest.mark.parametrize('configured', ['0', '-1'])
def test_pool_worker_nofile_rejects_nonpositive_values(configured):
    result = _run_nofile_probe(f"""
        import json
        import os

        os.environ['POOL_NOFILE_SOFT_LIMIT'] = {configured!r}
        from pool.worker_main import _configure_nofile_soft_limit

        try:
            _configure_nofile_soft_limit()
        except ValueError as exc:
            print(json.dumps({{'error': str(exc)}}))
        else:
            raise AssertionError('nonpositive nofile limit was accepted')
    """)

    assert result == {
        'error': 'POOL_NOFILE_SOFT_LIMIT must be greater than zero',
    }


def test_discover_activities_finds_all_snapshot_methods():
    instance = SnapshotActivities(
        pool_worker_uid='test-uid',
        controller=MagicMock(),
        storage_mode=PoolStorageMode.DAEMON_ON_PVC,
        gpu_concurrency=1,
    )
    discovered = _discover_activities(instance)
    discovered_names = sorted(m.__name__ for m in discovered)
    assert discovered_names == [
        'build_ttng_custom_pred_table_bundle',
        'build_ttng_custom_train_table_bundle',
        'delete_snapshot',
        'ensure_snapshot',
        'gen_tt',
        'materialize_prediction_task',
        'materialize_training_task',
        'predict',
        'train',
    ]


def test_discover_activities_skips_private_helpers():
    instance = SnapshotActivities(
        pool_worker_uid='test-uid',
        controller=MagicMock(),
        storage_mode=PoolStorageMode.DAEMON_ON_PVC,
        gpu_concurrency=1,
    )
    discovered = _discover_activities(instance)
    names = {m.__name__ for m in discovered}
    assert '_ensure_snapshot' not in names
    assert 'start' not in names
    assert 'stop' not in names


def test_shutdown_grace_is_immediate_only_for_long_lived_daemon():
    assert _SIGNAL_SHUTDOWN_TIMEOUT_SECS < 30
    assert 0 < _ACTIVITY_SHUTDOWN_GRACE_SECS < _SIGNAL_SHUTDOWN_TIMEOUT_SECS
    assert _graceful_shutdown_timeout(
        PoolStorageMode.DAEMON_ON_PVC) == timedelta()
    assert _graceful_shutdown_timeout(
        PoolStorageMode.ACTIVITY_ON_PVC) == timedelta(
            seconds=_ACTIVITY_SHUTDOWN_GRACE_SECS)
    assert _graceful_shutdown_timeout(
        PoolStorageMode.ACTIVITY_ON_EMPTYDIR) == timedelta(
            seconds=_ACTIVITY_SHUTDOWN_GRACE_SECS)


@pytest.mark.asyncio
@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
async def test_worker_signal_requests_orderly_shutdown(monkeypatch, signum):
    handlers, removed = _capture_signal_handlers(monkeypatch)
    worker = _FakeWorker()

    run_task = asyncio.create_task(_run_worker_until_shutdown(worker))
    await worker.started.wait()
    callback, args = handlers[signum]
    callback(*args)
    signal_shutdown = await asyncio.wait_for(run_task, timeout=1)

    assert worker.shutdown_calls == 1
    assert removed == [signal.SIGTERM, signal.SIGINT]
    assert signal_shutdown is True


@pytest.mark.asyncio
async def test_repeated_worker_signal_requests_shutdown_once(monkeypatch):
    handlers, _ = _capture_signal_handlers(monkeypatch)
    worker = _FakeWorker()

    run_task = asyncio.create_task(_run_worker_until_shutdown(worker))
    await worker.started.wait()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGTERM):
        callback, args = handlers[signum]
        callback(*args)
    signal_shutdown = await asyncio.wait_for(run_task, timeout=1)

    assert worker.shutdown_calls == 1
    assert signal_shutdown is True


@pytest.mark.asyncio
async def test_signal_wins_when_worker_and_signal_finish_together(monkeypatch):
    handlers, _ = _capture_signal_handlers(monkeypatch)
    worker = _FakeWorker()

    run_task = asyncio.create_task(_run_worker_until_shutdown(worker))
    await worker.started.wait()
    worker.complete.set()
    callback, args = handlers[signal.SIGTERM]
    callback(*args)
    signal_shutdown = await asyncio.wait_for(run_task, timeout=1)

    assert signal_shutdown is True
    assert worker.shutdown_calls == 0


@pytest.mark.asyncio
async def test_signal_after_activity_done_still_forces_signal_exit(monkeypatch):
    handlers, _ = _capture_signal_handlers(monkeypatch)

    class _BlockingShutdownWorker(_FakeWorker):
        def __init__(self):
            super().__init__()
            self.shutdown_started = asyncio.Event()
            self.allow_shutdown = asyncio.Event()

        async def shutdown(self):
            self.shutdown_calls += 1
            self.shutdown_started.set()
            await self.allow_shutdown.wait()
            self.complete.set()

    worker = _BlockingShutdownWorker()
    activity_done = asyncio.Event()
    run_task = asyncio.create_task(
        _run_worker_until_shutdown(worker, activity_done=activity_done))
    await worker.started.wait()
    activity_done.set()
    await worker.shutdown_started.wait()

    callback, args = handlers[signal.SIGTERM]
    callback(*args)
    worker.allow_shutdown.set()
    signal_shutdown = await asyncio.wait_for(run_task, timeout=1)

    assert signal_shutdown is True
    assert worker.shutdown_calls == 1


@pytest.mark.asyncio
async def test_signal_shutdown_deadline_is_fail_closed(monkeypatch):
    handlers, _ = _capture_signal_handlers(monkeypatch)
    signal_requested = asyncio.Event()

    class _StuckShutdownWorker(_FakeWorker):
        async def shutdown(self):
            self.shutdown_calls += 1
            await asyncio.Event().wait()

    monkeypatch.setattr('pool.worker_main._SIGNAL_SHUTDOWN_TIMEOUT_SECS', 0.01)
    worker = _StuckShutdownWorker()
    run_task = asyncio.create_task(
        _run_worker_until_shutdown(
            worker,
            signal_requested=signal_requested,
        ))
    await worker.started.wait()
    callback, args = handlers[signal.SIGTERM]
    callback(*args)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(run_task, timeout=1)
    assert worker.shutdown_calls == 1
    assert signal_requested.is_set()


@pytest.mark.asyncio
async def test_activity_done_uses_same_orderly_shutdown_path(monkeypatch):
    _capture_signal_handlers(monkeypatch)
    worker = _FakeWorker()
    activity_done = asyncio.Event()

    run_task = asyncio.create_task(
        _run_worker_until_shutdown(worker, activity_done=activity_done))
    await worker.started.wait()
    activity_done.set()
    signal_shutdown = await asyncio.wait_for(run_task, timeout=1)

    assert worker.shutdown_calls == 1
    assert signal_shutdown is False


def test_signal_exit_does_not_join_default_executor(tmp_path):
    """A native call abandoned after worker shutdown cannot keep PID 1 alive."""
    output = tmp_path / 'native-side-effects'
    program = f'''\
import asyncio
import os
import time
from pool.worker_main import _exit_after_signal_shutdown

output = {str(output)!r}

def native_work():
    with open(output, 'ab', buffering=0) as stream:
        stream.write(b'started\\n')
        time.sleep(5)
        stream.write(b'published\\n')

async def main():
    asyncio.create_task(asyncio.to_thread(native_work))
    for _ in range(100):
        if os.path.exists(output):
            break
        await asyncio.sleep(0.01)
    assert os.path.exists(output)
    _exit_after_signal_shutdown()

asyncio.run(main())
'''
    process = subprocess.Popen(
        [sys.executable, '-c', program],
    )
    for _ in range(3000):
        if output.exists():
            break
        time.sleep(0.01)
    assert output.exists()
    started = time.monotonic()
    returncode = process.wait(timeout=2)
    elapsed = time.monotonic() - started
    output_after_exit = output.read_bytes()
    time.sleep(0.1)

    assert returncode == 0
    assert elapsed < 2
    assert output_after_exit == b'started\n'
    assert output.read_bytes() == output_after_exit


def test_signal_shutdown_timeout_exits_without_joining_default_executor(
        tmp_path):
    """A failed signal shutdown still exits nonzero before executor join."""
    output = tmp_path / 'native-timeout-side-effects'
    ready = tmp_path / 'worker-ready'
    program = f'''\
import asyncio
import os
import time
import pool.worker_main as worker_main
from pool.worker_main import _finalize_worker, _run_worker_until_shutdown

output = {str(output)!r}
ready = {str(ready)!r}
worker_main._SIGNAL_SHUTDOWN_TIMEOUT_SECS = 0.05

def native_work():
    with open(output, 'ab', buffering=0) as stream:
        stream.write(b'started\\n')
        time.sleep(5)
        stream.write(b'published\\n')

class StuckWorker:
    async def run(self):
        asyncio.create_task(asyncio.to_thread(native_work))
        while not os.path.exists(output):
            await asyncio.sleep(0.01)
        open(ready, 'wb').close()
        await asyncio.Event().wait()

    async def shutdown(self):
        await asyncio.Event().wait()

class WorkerEnv:
    def shutdown(self):
        pass

class Activities:
    async def stop(self):
        pass

async def main():
    signal_requested = asyncio.Event()
    signal_shutdown = False
    signal_shutdown_failed = False
    try:
        signal_shutdown = await _run_worker_until_shutdown(
            StuckWorker(), signal_requested=signal_requested)
    except BaseException:
        signal_shutdown_failed = signal_requested.is_set()
        raise
    finally:
        await _finalize_worker(
            WorkerEnv(),
            Activities(),
            signal_shutdown=(signal_shutdown or signal_requested.is_set()),
            signal_shutdown_failed=signal_shutdown_failed,
        )

asyncio.run(main())
'''
    process = subprocess.Popen([sys.executable, '-c', program])
    for _ in range(3000):
        if ready.exists():
            break
        time.sleep(0.01)
    assert ready.exists()

    started = time.monotonic()
    process.send_signal(signal.SIGTERM)
    returncode = process.wait(timeout=2)
    elapsed = time.monotonic() - started
    output_after_exit = output.read_bytes()
    time.sleep(0.1)

    assert returncode == 1
    assert elapsed < 2
    assert output_after_exit == b'started\n'
    assert output.read_bytes() == output_after_exit


@pytest.mark.asyncio
@pytest.mark.parametrize('signal_shutdown', [False, True])
async def test_worker_resources_close_before_signal_exit(
        monkeypatch, signal_shutdown):
    calls = []
    worker_env = MagicMock()
    worker_env.shutdown.side_effect = lambda: calls.append('worker-env')
    activities = MagicMock()

    async def _stop():
        calls.append('activities')

    activities.stop = _stop
    monkeypatch.setattr(
        'pool.worker_main._exit_after_signal_shutdown',
        lambda exit_code: calls.append(f'exit-{exit_code}'),
    )

    await _finalize_worker(
        worker_env,
        activities,
        signal_shutdown=signal_shutdown,
    )

    expected = ['worker-env', 'activities']
    if signal_shutdown:
        expected.append('exit-0')
    assert calls == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_step', ['worker-env', 'activities'])
@pytest.mark.parametrize('signal_shutdown', [False, True])
async def test_worker_cleanup_failure_attempts_both_and_fails_closed(
        monkeypatch, failed_step, signal_shutdown):
    calls = []
    error = RuntimeError(f'{failed_step} failed')
    worker_env = MagicMock()

    def _shutdown():
        calls.append('worker-env')
        if failed_step == 'worker-env':
            raise error

    worker_env.shutdown.side_effect = _shutdown
    activities = MagicMock()

    async def _stop():
        calls.append('activities')
        if failed_step == 'activities':
            raise error

    activities.stop = _stop
    monkeypatch.setattr(
        'pool.worker_main._exit_after_signal_shutdown',
        lambda exit_code: calls.append(f'exit-{exit_code}'),
    )

    if signal_shutdown:
        await _finalize_worker(
            worker_env,
            activities,
            signal_shutdown=True,
        )
        assert calls == ['worker-env', 'activities', 'exit-1']
    else:
        with pytest.raises(RuntimeError, match=f'{failed_step} failed'):
            await _finalize_worker(
                worker_env,
                activities,
                signal_shutdown=False,
            )
        assert calls == ['worker-env', 'activities']


@pytest.mark.asyncio
async def test_worker_failure_remains_primary(monkeypatch):
    _capture_signal_handlers(monkeypatch)
    worker = _FakeWorker(failure=RuntimeError('worker failed'))

    run_task = asyncio.create_task(_run_worker_until_shutdown(worker))
    await worker.started.wait()
    worker.complete.set()
    with pytest.raises(RuntimeError, match='worker failed'):
        await run_task

    assert worker.shutdown_calls == 0


@pytest.mark.asyncio
async def test_normal_worker_completion_is_not_signal_shutdown(monkeypatch):
    _capture_signal_handlers(monkeypatch)
    worker = _FakeWorker()

    run_task = asyncio.create_task(_run_worker_until_shutdown(worker))
    await worker.started.wait()
    worker.complete.set()
    signal_shutdown = await asyncio.wait_for(run_task, timeout=1)

    assert signal_shutdown is False
    assert worker.shutdown_calls == 0


@pytest.mark.asyncio
async def test_signal_handler_installation_failure_is_fail_closed(monkeypatch):
    loop = asyncio.get_running_loop()
    removed = []

    def _add(signum, _callback, *_args):
        if signum == signal.SIGINT:
            raise RuntimeError('handler unavailable')

    monkeypatch.setattr(loop, 'add_signal_handler', _add)
    monkeypatch.setattr(loop, 'remove_signal_handler',
                        lambda signum: removed.append(signum) or True)
    worker = _FakeWorker()

    with pytest.raises(RuntimeError, match='handler unavailable'):
        await _run_worker_until_shutdown(worker)

    assert removed == [signal.SIGTERM]
    assert worker.started.is_set() is False


@pytest.mark.skipif(
    os.getpid() != 1 or not os.getenv('KUMO_TEST_TEMPORAL_ADDRESS'),
    reason='requires Python PID 1 and an isolated Temporal test server',
)
@pytest.mark.asyncio
@pytest.mark.parametrize('storage_mode,ignore_term,max_shutdown_seconds', [
    (PoolStorageMode.DAEMON_ON_PVC, False, 10.0),
    (PoolStorageMode.ACTIVITY_ON_PVC, False, 10.0),
    (PoolStorageMode.DAEMON_ON_PVC, True, 30.0),
    (PoolStorageMode.ACTIVITY_ON_PVC, True, 30.0),
])
async def test_real_temporal_worker_sigterm_cancels_activity_before_pod_grace(
        tmp_path, monkeypatch, storage_mode, ignore_term,
        max_shutdown_seconds):
    """SDK 1.30 shutdown cancels and reaps before container/pod grace."""
    if not _subprocess_supervisor_capabilities_available():
        pytest.skip('strong pidfd/procfs supervisor unavailable')

    child_pid_path = tmp_path / 'real-temporal-child.pid'
    output_path = tmp_path / 'real-temporal-output.log'
    command = tmp_path / 'kumo-pred'
    child_term_setup = (
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        if ignore_term else '')
    command.write_text(f'''\
#!{sys.executable}
import os
import signal
import subprocess
import sys
import time

output = open({str(output_path)!r}, 'ab', buffering=0)
child = subprocess.Popen([
    sys.executable,
    '-c',
    'import os, signal, sys, time; '
    {child_term_setup!r}
    'os.write(int(sys.argv[1]), b"child-ready\\n"); time.sleep(60)',
    str(output.fileno()),
], stdout=output, stderr=output, pass_fds=(output.fileno(),))
import os as _os
with open({str(child_pid_path)!r} + '.tmp', 'w') as pid_file:
    pid_file.write(str(child.pid))
_os.replace({str(child_pid_path)!r} + '.tmp', {str(child_pid_path)!r})
print('parent-ready', flush=True)
time.sleep(60)
''')
    command.chmod(0o755)

    real_exists = os.path.exists
    monkeypatch.setattr(
        'pool.activities.osp.exists',
        lambda path: (False if str(path).endswith('/bin/kumo-pred') else
                      real_exists(path)))
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')

    client = await Client.connect(os.environ['KUMO_TEST_TEMPORAL_ADDRESS'])
    task_queue = f'pid1-shutdown-{uuid.uuid4()}'
    worker = Worker(
        client,
        task_queue=task_queue,
        activities=[_pid1_native_shutdown_activity],
        workflows=[_PID1NativeShutdownWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        graceful_shutdown_timeout=_graceful_shutdown_timeout(storage_mode),
    )
    worker_task = asyncio.create_task(_run_worker_until_shutdown(worker))
    handle = await client.start_workflow(
        _PID1NativeShutdownWorkflow.run,
        'marker',
        id=f'pid1-shutdown-{uuid.uuid4()}',
        task_queue=task_queue,
    )
    for _ in range(1000):
        if (child_pid_path.exists() and output_path.exists()
                and b'child-ready' in output_path.read_bytes()):
            break
        await asyncio.sleep(0.01)
    assert (child_pid_path.exists() and output_path.exists()
            and b'child-ready' in output_path.read_bytes()), (
                'real Temporal activity never started its native descendant')
    child_pid = int(child_pid_path.read_text())

    shutdown_started = time.monotonic()
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(worker_task, timeout=max_shutdown_seconds)
    shutdown_elapsed = time.monotonic() - shutdown_started
    with contextlib.suppress(Exception):
        await handle.cancel()

    assert shutdown_elapsed < max_shutdown_seconds
    assert not os.path.exists(f'/proc/{child_pid}')


@pytest.mark.skipif(
    os.getpid() != 1 or not os.getenv('KUMO_TEST_TEMPORAL_ADDRESS'),
    reason='requires Python PID 1 and an isolated Temporal test server',
)
@pytest.mark.asyncio
async def test_activity_worker_grace_preserves_terminal_result():
    """The bounded activity grace still lets a completed result reach Core."""
    global _activity_result_done
    _activity_result_done = asyncio.Event()
    client = await Client.connect(os.environ['KUMO_TEST_TEMPORAL_ADDRESS'])
    task_queue = f'pid1-result-{uuid.uuid4()}'
    activity_worker = Worker(
        client,
        task_queue=task_queue,
        activities=[_pid1_activity_result],
        graceful_shutdown_timeout=_graceful_shutdown_timeout(
            PoolStorageMode.ACTIVITY_ON_PVC),
    )
    workflow_worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[_PID1ActivityResultWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    activity_worker_task = asyncio.create_task(
        _run_worker_until_shutdown(activity_worker,
                                   activity_done=_activity_result_done))
    workflow_worker_task = asyncio.create_task(workflow_worker.run())
    result = await client.execute_workflow(
        _PID1ActivityResultWorkflow.run,
        'reported',
        id=f'pid1-result-{uuid.uuid4()}',
        task_queue=task_queue,
    )
    await asyncio.wait_for(activity_worker_task, timeout=10)
    await workflow_worker.shutdown()
    await workflow_worker_task

    assert result == 'reported'


@pytest.mark.skipif(os.getpid() != 1,
                    reason='requires production-like Python PID 1')
@pytest.mark.asyncio
@pytest.mark.parametrize('ignore_term,max_shutdown_seconds', [
    (False, 10.0),
    (True, 30.0),
])
async def test_pid1_sigterm_cancels_and_reaps_native_subprocess(
        tmp_path, monkeypatch, ignore_term, max_shutdown_seconds):
    """Container TERM reaches Temporal shutdown and the native cleanup path."""
    if not _subprocess_supervisor_capabilities_available():
        pytest.skip('strong pidfd/procfs supervisor unavailable')

    child_pid_path = tmp_path / 'native-child.pid'
    output_path = tmp_path / 'native-output.log'
    command = tmp_path / 'kumo-pred'
    child_term_setup = (
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        if ignore_term else '')
    command.write_text(f'''\
#!{sys.executable}
import os
import signal
import subprocess
import sys
import time

output = open({str(output_path)!r}, 'ab', buffering=0)
child = subprocess.Popen([
    sys.executable,
    '-c',
    'import os, signal, sys, time; '
    {child_term_setup!r}
    'os.write(int(sys.argv[1]), b"child-ready\\n"); time.sleep(60)',
    str(output.fileno()),
], stdout=output, stderr=output, pass_fds=(output.fileno(),))
import os as _os
with open({str(child_pid_path)!r} + '.tmp', 'w') as pid_file:
    pid_file.write(str(child.pid))
_os.replace({str(child_pid_path)!r} + '.tmp', {str(child_pid_path)!r})
print('parent-ready', flush=True)
time.sleep(60)
''')
    command.chmod(0o755)

    real_exists = os.path.exists
    monkeypatch.setattr(
        'pool.activities.osp.exists',
        lambda path: (False if str(path).endswith('/bin/kumo-pred') else
                      real_exists(path)))
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    monkeypatch.setattr('pool.activities.activity.heartbeat', MagicMock())
    monkeypatch.setattr('pool.activities._SUBPROCESS_EXIT_POLL_SECS', 0.01)

    class _ActivityWorker:
        def __init__(self):
            self.activity_task = None
            self.shutdown_started = False
            self.shutdown_calls = 0

        async def run(self):
            self.activity_task = asyncio.create_task(
                _run_kumo_subprocess('kumo-pred', 'pid1-term',
                                     '/tmp/unused.json', None))
            try:
                await self.activity_task
            except asyncio.CancelledError:
                if not self.shutdown_started:
                    raise

        async def shutdown(self):
            self.shutdown_calls += 1
            self.shutdown_started = True
            assert self.activity_task is not None
            self.activity_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.activity_task

    worker = _ActivityWorker()
    worker_task = asyncio.create_task(_run_worker_until_shutdown(worker))
    for _ in range(500):
        if (child_pid_path.exists() and output_path.exists()
                and b'child-ready' in output_path.read_bytes()):
            break
        await asyncio.sleep(0.01)
    assert (child_pid_path.exists() and output_path.exists()
            and b'child-ready' in output_path.read_bytes()), (
                'native descendant never became signal-ready')
    child_pid = int(child_pid_path.read_text())

    shutdown_started = time.monotonic()
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(worker_task, timeout=max_shutdown_seconds)
    shutdown_elapsed = time.monotonic() - shutdown_started
    output_after_shutdown = output_path.read_bytes()
    await asyncio.sleep(0.1)

    assert worker.shutdown_calls == 1
    assert shutdown_elapsed < max_shutdown_seconds
    assert not os.path.exists(f'/proc/{child_pid}')
    assert output_path.read_bytes() == output_after_shutdown
    output_stat = output_path.stat()
    holders = []
    for proc_fd_dir in os.scandir('/proc'):
        if not proc_fd_dir.name.isdigit():
            continue
        try:
            fd_entries = os.scandir(f'{proc_fd_dir.path}/fd')
        except (FileNotFoundError, PermissionError):
            continue
        with fd_entries:
            for fd_entry in fd_entries:
                try:
                    fd_stat = fd_entry.stat(follow_symlinks=True)
                except (FileNotFoundError, PermissionError):
                    continue
                if (fd_stat.st_dev == output_stat.st_dev
                        and fd_stat.st_ino == output_stat.st_ino):
                    holders.append(f'{proc_fd_dir.name}/{fd_entry.name}')
    assert holders == []
