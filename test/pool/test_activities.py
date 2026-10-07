# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pool.activities``.

Real activity-body behavior is covered by the diskgraph e2e
integration tests.  Here we lock in the structural contract the
worker entrypoint depends on:

- ``SnapshotActivities`` constructs cleanly with a fake controller.
- Lifecycle (``start`` / ``stop``) is idempotent and respects storage mode.
- Module-level ``place_snapshot_activity`` / ``release_snapshot_activity``
  are valid ``@activity.defn`` callables (so workflow workers can
  register them).

Plus pool-specific behavior tests:

- Subprocess cancellation propagation — workflow cancellation must
  terminate kumo-auto / kumo-pred so they don't keep running for
  hours after the activity is cancelled (see
  ``_run_kumo_subprocess``).
- GPU semaphore serialization — concurrent train/predict on the same
  worker must be serialized by the per-instance semaphore so a single
  GPU isn't fought over.
"""
import asyncio
import contextlib
import contextvars
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path
from test.pool.prediction_materialization_lock_helper import (
    hold_prediction_extension_shared_lock,
    hold_prediction_materialization_ownership,
)
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock as _MockController

import pytest
from temporalio import activity

import pool.activities as activities
from io_utils.json import load_json
from kumo.diskgraph.job_sidecars import (
    cleanup_local_job_sidecars_from_config_path,
    restore_job_sidecars_for_use_from_config_path,
    restore_job_sidecars_from_config_path,
    upload_job_sidecars_from_config_path,
)
from pool.activities import (
    _SUBPROCESS_OUTPUT_MAX_RECORD_BYTES,
    IN_PROGRESS_MARKER,
    SUCCESS_MARKER,
    SnapshotActivities,
    _clean_in_progress_snapshots,
    _drain_subprocess_output,
    _GpuSlots,
    _iter_local_snapshot_dirs,
    _launch_owned_subprocess,
    _materialize_and_upload_prediction_task,
    _OwnedSubprocessSession,
    _ProcessIdentity,
    _read_process_identity,
    _run_diskgraph_async,
    _run_kumo_subprocess,
    _schedule_activity_heartbeat,
    _shard_pred_config,
    _subprocess_supervisor_capabilities_available,
    _upload_prediction_artifact_to_databricks,
    place_snapshot_activity,
    release_snapshot_activity,
)
from pool.dataclasses import WorkerEvent, WorkerEventType
from pool.locks import PerSnapshotLocks
from pool.storage_mode import PoolStorageMode
from pool.training_progress import (
    TRAINING_PROGRESS_PUBLISH_TIMEOUT_SECS,
    publish_training_progress,
    training_progress_output,
)
from rest.backends.ml_backend.ml_job_output_store import MLJobOutputStore
from rest.config.ml_backend import JobState
from rest.config.training_job import TrainingJobProgress
from temporalio_commons.error_utils import is_error_non_retryable


class _DiskgraphCorruptionError(RuntimeError):
    """Local stand-in for the optional native diskgraph exception."""


_FAKE_DISKGRAPH = SimpleNamespace(
    DiskgraphCorruptionError=_DiskgraphCorruptionError)


def test_process_identity_detects_live_sibling_behind_zombie_leader() -> None:
    identity = _ProcessIdentity(
        pid=100,
        process_group_id=100,
        session_id=100,
        start_time_ticks=5,
        state='Z',
        proc_device=1,
        proc_inode=2,
        pid_namespace_device=3,
        pid_namespace_inode=4,
        task_states=((100, 'Z'), (101, 'R')),
    )

    assert identity.has_live_tasks


def test_process_identity_falls_back_to_live_leader_for_empty_task_race(
) -> None:
    identity = _ProcessIdentity(
        pid=100,
        process_group_id=100,
        session_id=100,
        start_time_ticks=5,
        state='S',
        proc_device=1,
        proc_inode=2,
        pid_namespace_device=3,
        pid_namespace_inode=4,
        task_states=(),
    )

    assert identity.has_live_tasks


@pytest.mark.parametrize(('task_group_exited', 'expected_live'), [
    (False, True),
    (None, True),
    (True, False),
])
def test_process_identity_empty_zombie_snapshot_uses_pidfd_group_state(
    task_group_exited: bool | None,
    expected_live: bool,
) -> None:
    identity = _ProcessIdentity(
        pid=100,
        process_group_id=100,
        session_id=100,
        start_time_ticks=5,
        state='Z',
        proc_device=1,
        proc_inode=2,
        pid_namespace_device=3,
        pid_namespace_inode=4,
        task_states=(),
        task_group_exited=task_group_exited,
    )

    assert identity.has_live_tasks is expected_live


@pytest.mark.parametrize(('task_group_exited', 'expected_live'), [
    (False, True),
    (None, True),
    (True, False),
])
def test_process_identity_uses_pidfd_when_task_state_is_unavailable(
    task_group_exited: bool | None,
    expected_live: bool,
) -> None:
    identity = _ProcessIdentity(
        pid=100,
        process_group_id=100,
        session_id=100,
        start_time_ticks=5,
        state='Z',
        proc_device=1,
        proc_inode=2,
        pid_namespace_device=3,
        pid_namespace_inode=4,
        task_states=None,
        task_group_exited=task_group_exited,
    )

    assert identity.has_live_tasks is expected_live


def test_read_process_task_states_distinguishes_unavailable_directory(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def _unavailable(_path: str) -> None:
        raise FileNotFoundError('synthetic unavailable task directory')

    monkeypatch.setattr(activities.os, 'scandir', _unavailable)

    assert activities._read_process_task_states(100) is None


@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='process identity uses Linux procfs and pidfds')
def test_read_process_identity_retains_live_pidfd_when_tasks_unavailable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(activities, '_read_process_task_states',
                        lambda _pid: None)
    monkeypatch.setattr(activities, 'pidfd_process_exited', lambda _pid: False)

    identity = activities._read_process_identity(os.getpid())

    assert identity.task_states is None
    assert identity.task_group_exited is False
    assert identity.has_live_tasks


def test_read_process_identity_uses_pidfd_for_empty_zombie_snapshot(
        monkeypatch: pytest.MonkeyPatch) -> None:
    proc_stat = SimpleNamespace(st_dev=1, st_ino=2)
    pid_namespace = SimpleNamespace(st_dev=3, st_ino=4)

    def _stat(path: str) -> SimpleNamespace:
        return pid_namespace if path.endswith('/ns/pid') else proc_stat

    pidfd_calls = []
    monkeypatch.setattr(activities.os, 'stat', _stat)
    monkeypatch.setattr(activities, '_read_process_task_states', lambda _pid:
                        ())
    monkeypatch.setattr(activities, 'pidfd_process_exited',
                        lambda pid: pidfd_calls.append(pid) or False)
    process_stat = b'100 (probe) Z 1 100 100 ' + b'0 ' * 15 + b'5'
    with mock.patch('builtins.open', mock.mock_open(read_data=process_stat)):
        identity = activities._read_process_identity(100)

    assert pidfd_calls == [100]
    assert identity.task_states == ()
    assert identity.task_group_exited is False
    assert identity.has_live_tasks


def _make_instance(
    storage_mode: PoolStorageMode = PoolStorageMode.DAEMON_ON_PVC
) -> SnapshotActivities:
    return SnapshotActivities(
        pool_worker_uid='test-uid',
        controller=_MockController(),
        storage_mode=storage_mode,
        gpu_concurrency=1,
    )


def _make_prediction_job_input():
    from kumo_core.types import PredictionArtifactType

    return SimpleNamespace(
        config=SimpleNamespace(outputs=[
            SimpleNamespace(artifact_type=PredictionArtifactType.PREDICTIONS)
        ]),
        output_connectors={},
    )


def _make_prediction_config(output_dir: str):
    return SimpleNamespace(
        output_dir=output_dir,
        start_partition_id=0,
        stop_partition_id=1,
        num_partitions=1,
        effective_output_num_partitions=1,
        diskgraph_task_dataset_bundle_dir='/tmp/bundle',
        return_embeddings=False,
    )


def test_uncancel_task_supports_python_310_tasks():
    class Python310Task:
        pass

    class ModernTask:
        calls = 0

        def uncancel(self):
            self.calls += 1

    activities._uncancel_task(Python310Task())
    modern_task = ModernTask()
    activities._uncancel_task(modern_task)
    assert modern_task.calls == 1


async def _wait_for_thread_event(event: threading.Event) -> None:
    for _ in range(1000):
        if event.is_set():
            return
        await asyncio.sleep(0.001)
    pytest.fail('worker-thread event was not set')


async def _wait_for_task_done(task: asyncio.Task) -> None:
    for _ in range(1000):
        if task.done():
            return
        await asyncio.sleep(0.001)
    pytest.fail('worker-thread task did not complete')


def _close_runner_while_diskgraph_call_is_active(
    target_path: str,
    connector_finished_path: str,
    runner_started_path: str,
    runner_closed_path: str,
) -> None:
    """Exercise the same all-task cancellation used by ``asyncio.run``."""
    target = Path(target_path)
    connector_finished = Path(connector_finished_path)
    runner_started = Path(runner_started_path)
    runner_closed = Path(runner_closed_path)
    started = threading.Event()
    release = threading.Event()

    def download() -> None:
        target.mkdir()
        (target / 'before-cancel.parquet').write_bytes(b'before')
        started.set()
        assert release.wait(timeout=5)
        (target / '_DOWNLOAD_SUCCESS').write_text('complete\n')
        connector_finished.write_text('complete\n')

    async def start_download() -> asyncio.Task:
        task = asyncio.create_task(
            _run_diskgraph_async(
                download,
                join_on_cancel=True,
                cancellation_cleanup_dir=str(target),
            ))
        assert await asyncio.to_thread(started.wait, 1)
        return task

    runner = asyncio.Runner()
    task = runner.run(start_download())

    def release_download() -> None:
        time.sleep(0.05)
        release.set()

    runner_started.write_text('complete\n')
    release_thread = threading.Thread(
        target=release_download,
        daemon=True,
    )
    release_thread.start()
    runner.close()
    release_thread.join(timeout=1)
    assert task.cancelled()
    runner_closed.write_text('complete\n')


def test_construct_with_mock_controller():
    instance = _make_instance()
    assert instance._uid == 'test-uid'
    assert instance.activity_done.is_set() is False


def test_schedule_activity_heartbeat_on_event_loop():
    activity_loop = mock.MagicMock()

    _schedule_activity_heartbeat(activity_loop)

    activity_loop.call_soon_threadsafe.assert_called_once_with(
        activity.heartbeat)


@pytest.mark.asyncio
async def test_diskgraph_cancellation_cleans_late_credentials(tmp_path):
    from async_utils.blockbuster_guard import (
        BlockBusterMode,
        _reset_blockbuster_guard_for_tests,
        activate_blockbuster_guard,
    )

    target = tmp_path / 'databricks-custom'
    target.mkdir()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def prepare():
        started.set()
        assert release.wait(timeout=5)
        (target / 'databricks.cfg').write_text('[profile]\ntoken = secret\n')
        finished.set()

    _reset_blockbuster_guard_for_tests()
    activate_blockbuster_guard(mode=BlockBusterMode.RAISE)
    try:
        task = asyncio.create_task(
            _run_diskgraph_async(prepare, abandonment_cleanup_dir=str(target)))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        for _ in range(100):
            if not await asyncio.to_thread(target.exists):
                break
            await asyncio.sleep(0.01)
        assert not await asyncio.to_thread(target.exists)
    finally:
        release.set()
        _reset_blockbuster_guard_for_tests()


def test_prediction_materialization_ownership_serializes_processes(tmp_path):
    context = multiprocessing.get_context('spawn')
    first_entered = context.Event()
    first_release = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    other_entered = context.Event()
    other_release = context.Event()

    first = context.Process(
        target=hold_prediction_materialization_ownership,
        args=(str(tmp_path), 'bp-job-1', first_entered, first_release),
    )
    second = context.Process(
        target=hold_prediction_materialization_ownership,
        args=(str(tmp_path), 'bp-job-1', second_entered, second_release),
    )
    other = context.Process(
        target=hold_prediction_materialization_ownership,
        args=(str(tmp_path), 'bp-job-2', other_entered, other_release),
    )
    try:
        first.start()
        assert first_entered.wait(timeout=30)
        second.start()
        other.start()
        assert other_entered.wait(timeout=30)
        assert not second_entered.wait(timeout=0.2)

        other_release.set()
        first_release.set()
        assert second_entered.wait(timeout=30)
        second_release.set()
        for process in (first, second, other):
            process.join(timeout=15)
    finally:
        for process in (first, second, other):
            if process.pid is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
    assert [first.exitcode, second.exitcode, other.exitcode] == [0, 0, 0]


def test_prediction_materialization_ownership_blocks_shared_consumer(tmp_path):
    context = multiprocessing.get_context('spawn')
    producer_entered = context.Event()
    producer_release = context.Event()
    reader_entered = context.Event()
    reader_release = context.Event()
    producer = context.Process(
        target=hold_prediction_materialization_ownership,
        args=(str(tmp_path), 'bp-job-1', producer_entered, producer_release),
    )
    reader = context.Process(
        target=hold_prediction_extension_shared_lock,
        args=(str(tmp_path), 'bp-job-1', reader_entered, reader_release),
    )
    try:
        producer.start()
        assert producer_entered.wait(timeout=30)
        reader.start()
        assert not reader_entered.wait(timeout=0.2)
        producer_release.set()
        assert reader_entered.wait(timeout=30)
        reader_release.set()
        for process in (producer, reader):
            process.join(timeout=15)
    finally:
        for process in (producer, reader):
            if process.pid is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
    assert [producer.exitcode, reader.exitcode] == [0, 0]


def test_prediction_materialization_holds_ownership_through_upload(tmp_path):
    events = []
    job_input = SimpleNamespace(id='bp-job-1')

    @contextlib.contextmanager
    def ownership(index_path, job_id):
        assert index_path == str(tmp_path)
        assert job_id == 'bp-job-1'
        events.append('lock')
        try:
            yield
        finally:
            events.append('unlock')

    def materialize(*_args):
        events.append('materialize')
        return ('pred-config.json', 'task', 'response')

    def upload(*_args, **_kwargs):
        events.append('upload')

    config = SimpleNamespace(diskgraph_index_path=str(tmp_path))
    with mock.patch('kumo_core.job_config.PredictionConfig.load',
                    return_value=config), \
            mock.patch('pool.activities.prediction_materialization_ownership',
                       ownership), \
            mock.patch('pool.activities.materialize_prediction_task',
                       materialize), \
            mock.patch('pool.activities.upload_job_sidecars_from_config_path',
                       upload):
        result = _materialize_and_upload_prediction_task(
            'graph-1@v1', job_input, 'settings', False, 'pred-config.json',
            None, 's3://persist')

    assert result == ('pred-config.json', 'task', 'response')
    assert events == ['lock', 'materialize', 'upload', 'unlock']


def test_prediction_materialization_upload_failure_releases_for_retry(tmp_path):
    job_input = SimpleNamespace(id='bp-job-1')
    config = SimpleNamespace(diskgraph_index_path=str(tmp_path))
    upload = mock.MagicMock(side_effect=[RuntimeError('upload failed'), None])
    materialize = mock.MagicMock(return_value=('pred-config.json', 'task',
                                               'response'))

    with mock.patch('kumo_core.job_config.PredictionConfig.load',
                    return_value=config), \
            mock.patch('pool.activities.materialize_prediction_task',
                       materialize), \
            mock.patch('pool.activities.upload_job_sidecars_from_config_path',
                       upload):
        with pytest.raises(RuntimeError, match='upload failed'):
            _materialize_and_upload_prediction_task('graph-1@v1', job_input,
                                                    'settings', False,
                                                    'pred-config.json', None,
                                                    's3://persist')
        result = _materialize_and_upload_prediction_task(
            'graph-1@v1', job_input, 'settings', False, 'pred-config.json',
            None, 's3://persist')

    assert result == ('pred-config.json', 'task', 'response')
    assert materialize.call_count == 2
    assert upload.call_count == 2


def test_prediction_materialization_missing_index_preserves_materializer_error(
):
    job_input = SimpleNamespace(id='bp-job-1')
    config = SimpleNamespace(diskgraph_index_path=None)
    precise_error = ValueError('missing exact DiskGraph snapshot binding')
    upload = mock.MagicMock()

    with mock.patch('kumo_core.job_config.PredictionConfig.load',
                    return_value=config), \
            mock.patch('pool.activities.materialize_prediction_task',
                       side_effect=precise_error), \
            mock.patch('pool.activities.upload_job_sidecars_from_config_path',
                       upload):
        with pytest.raises(ValueError) as raised:
            _materialize_and_upload_prediction_task('graph-1@v1', job_input,
                                                    'settings', False,
                                                    'pred-config.json', None,
                                                    's3://persist')

    assert raised.value is precise_error
    upload.assert_not_called()


def test_prediction_materialization_missing_index_cannot_succeed_unpublished():
    job_input = SimpleNamespace(id='bp-job-1')
    config = SimpleNamespace(diskgraph_index_path=None)
    upload = mock.MagicMock()

    with mock.patch('kumo_core.job_config.PredictionConfig.load',
                    return_value=config), \
            mock.patch('pool.activities.materialize_prediction_task',
                       return_value=('pred-config.json', 'task', 'response')), \
            mock.patch('pool.activities.upload_job_sidecars_from_config_path',
                       upload):
        with pytest.raises(ValueError, match='cannot be safely published'):
            _materialize_and_upload_prediction_task('graph-1@v1', job_input,
                                                    'settings', False,
                                                    'pred-config.json', None,
                                                    's3://persist')

    upload.assert_not_called()


def test_diskgraph_runner_shutdown_joins_writer_and_cleans_exact_tree(tmp_path):
    """Runner-wide task cancellation must not poison the off-loop join."""
    target = tmp_path / 'snapshot-download'
    connector_finished = tmp_path / 'connector-finished'
    runner_started = tmp_path / 'runner-started'
    runner_closed = tmp_path / 'runner-closed'
    context = multiprocessing.get_context('spawn')
    process = context.Process(
        target=_close_runner_while_diskgraph_call_is_active,
        args=(
            str(target),
            str(connector_finished),
            str(runner_started),
            str(runner_closed),
        ),
    )
    process.start()

    startup_deadline = time.monotonic() + 20
    while not runner_started.exists() and time.monotonic() < startup_deadline:
        if not process.is_alive():
            break
        time.sleep(0.01)
    assert runner_started.exists(), (
        f'child exited {process.exitcode} before starting the runner')

    process.join(timeout=3)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        pytest.fail('runner shutdown spun after connector release '
                    f'(connector_finished={connector_finished.exists()})')

    assert process.exitcode == 0
    assert connector_finished.read_text() == 'complete\n'
    assert runner_closed.read_text() == 'complete\n'
    assert not target.exists()


@pytest.mark.asyncio
async def test_diskgraph_off_loop_dispatch_preserves_contextvars():
    request_id = contextvars.ContextVar('request_id')
    request_id.set('activity-request-id')

    assert await _run_diskgraph_async(request_id.get) == 'activity-request-id'


@pytest.mark.asyncio
async def test_diskgraph_cancellation_joins_download_and_removes_exact_tree(
        tmp_path):
    """Cancellation waits for a late writer, then removes its exact root."""
    target = tmp_path / 'snapshot-download'
    sibling = tmp_path / 'other-snapshot-download'
    sibling.mkdir()
    (sibling / 'owned-by-other-snapshot').write_bytes(b'keep')
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def download():
        target.mkdir()
        (target / 'before-cancel.parquet').write_bytes(b'before')
        started.set()
        assert release.wait(timeout=5)
        target.mkdir(exist_ok=True)
        (target / 'after-cancel.parquet').write_bytes(b'after')
        finished.set()

    task = asyncio.create_task(
        _run_diskgraph_async(
            download,
            join_on_cancel=True,
            cancellation_cleanup_dir=str(target),
        ))
    assert await asyncio.to_thread(started.wait, 1)
    cancelled_at = time.monotonic()
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()

    # Model eager workflow cleanup racing the still-running connector. The
    # connector recreates this tree after it resumes.
    shutil.rmtree(target)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    assert time.monotonic() - cancelled_at >= 0.05
    assert not target.exists()
    assert (sibling / 'owned-by-other-snapshot').read_bytes() == b'keep'


@pytest.mark.asyncio
async def test_cancelled_download_holds_same_snapshot_lock_until_cleanup(
        tmp_path):
    locks = PerSnapshotLocks()
    target = tmp_path / 'snapshot-download'
    other_target = tmp_path / 'other-snapshot-download'
    started = threading.Event()
    release = threading.Event()
    same_entered = asyncio.Event()
    other_entered = asyncio.Event()

    def download():
        target.mkdir()
        (target / 'old-attempt').write_bytes(b'old')
        started.set()
        assert release.wait(timeout=5)
        target.mkdir(exist_ok=True)
        (target / 'late-old-attempt').write_bytes(b'old')

    async def old_attempt():
        async with locks.acquire('snapshot'):
            await _run_diskgraph_async(
                download,
                join_on_cancel=True,
                cancellation_cleanup_dir=str(target),
            )

    async def same_snapshot_retry():
        async with locks.acquire('snapshot'):
            same_entered.set()
            target.mkdir()
            (target / 'new-attempt').write_bytes(b'new')

    async def other_snapshot():
        async with locks.acquire('other-snapshot'):
            other_entered.set()
            other_target.mkdir()

    old_task = asyncio.create_task(old_attempt())
    assert await asyncio.to_thread(started.wait, 1)
    old_task.cancel()
    same_task = asyncio.create_task(same_snapshot_retry())
    other_task = asyncio.create_task(other_snapshot())
    await asyncio.wait_for(other_entered.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert not same_entered.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await old_task
    await asyncio.wait_for(same_task, timeout=1)
    await asyncio.wait_for(other_task, timeout=1)

    # There is no late callback that can delete the retry's new tree after it
    # acquires the lock.
    await asyncio.sleep(0.05)
    assert (target / 'new-attempt').read_bytes() == b'new'
    assert not (target / 'late-old-attempt').exists()
    assert other_target.is_dir()


@pytest.mark.asyncio
async def test_cancelled_download_exception_is_logged_but_outcome_is_cancelled(
        tmp_path, caplog):
    target = tmp_path / 'snapshot-download'
    started = threading.Event()
    release = threading.Event()

    def download():
        target.mkdir()
        started.set()
        assert release.wait(timeout=5)
        (target / 'partial').write_bytes(b'partial')
        raise RuntimeError('late connector failure')

    task = asyncio.create_task(
        _run_diskgraph_async(
            download,
            join_on_cancel=True,
            cancellation_cleanup_dir=str(target),
        ))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    release.set()

    with caplog.at_level('WARNING'):
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not target.exists()
    assert any('failed after its caller was cancelled' in record.getMessage()
               for record in caplog.records)


def test_clean_in_progress_snapshots_removes_non_warm_dirs(tmp_path):
    warm = tmp_path / 'NA-11111111111111111111111111111111' / 'v1'
    warm.mkdir(parents=True)
    (warm / SUCCESS_MARKER).write_text('ok\n')

    partial = tmp_path / 'NA-22222222222222222222222222222222' / 'v2'
    partial.mkdir(parents=True)
    (partial / IN_PROGRESS_MARKER).write_text('')

    flat_partial = tmp_path / 'NA-33333333333333333333333333333333@v3'
    flat_partial.mkdir()
    (flat_partial / IN_PROGRESS_MARKER).write_text('')

    invalid_flat = tmp_path / 'foo@bar'
    invalid_flat.mkdir()
    (invalid_flat / IN_PROGRESS_MARKER).write_text('')

    non_snapshot_cache = tmp_path / 'pred_artifact_cache' / 'job-1'
    non_snapshot_cache.mkdir(parents=True)
    (non_snapshot_cache / 'artifact').write_text('keep')

    cleaned = _clean_in_progress_snapshots(str(tmp_path))

    assert cleaned == [
        'graph-22222222222222222222222222222222@v2',
        'graph-33333333333333333333333333333333@v3',
    ]
    assert warm.exists()
    assert not partial.exists()
    assert not partial.parent.exists()
    assert not flat_partial.exists()
    assert invalid_flat.exists()
    assert non_snapshot_cache.exists()


def test_iter_local_snapshot_dirs_skips_unreadable_lost_found(
        tmp_path, monkeypatch):
    # Regression for the diskgraph worker daemon CrashLoopBackOff on ext4
    # PVCs (e.g. Azure managed disks): the mount root carries a root-owned,
    # 0700 ``lost+found`` the non-root worker cannot ``scandir``. The scan
    # must skip it and still discover real snapshots rather than raising.
    warm = tmp_path / 'NA-44444444444444444444444444444444' / 'v4'
    warm.mkdir(parents=True)
    (warm / SUCCESS_MARKER).write_text('ok\n')

    lost_found = tmp_path / 'lost+found'
    lost_found.mkdir()

    real_iterdir = Path.iterdir

    def _fake_iterdir(self):
        # Emulate the unreadable ext4 lost+found on any test user.
        if self.name == 'lost+found':
            raise PermissionError(13, 'Permission denied', str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, 'iterdir', _fake_iterdir)

    snapshots = _iter_local_snapshot_dirs(str(tmp_path))

    assert snapshots == [
        ('graph-44444444444444444444444444444444@v4', warm),
    ]


def test_long_lived_start_creates_heartbeat_task():
    """DAEMON_ON_PVC workers spawn a background periodic-heartbeat task
    (one initial heartbeat + the loop)."""
    async def run() -> None:
        instance = _make_instance(PoolStorageMode.DAEMON_ON_PVC)

        def clean_in_progress(_root):
            with pytest.raises(RuntimeError, match='no running event loop'):
                asyncio.get_running_loop()
            return []

        with (
                mock.patch('pool.activities._clean_in_progress_snapshots',
                           side_effect=clean_in_progress),
                mock.patch('pool.activities.get_service') as get_service,
                mock.patch('pool.activities._probe_local_disks',
                           return_value=((0, 0, None), (0, 0, None))),
        ):
            get_service.return_value.list_ready_handles.coroutine = (
                mock.AsyncMock(return_value=[]))
            await instance.start()
            assert instance._heartbeat_task is not None
            assert not instance._heartbeat_task.done()
            await instance.stop()
            # stop() cancels + awaits the task.
            assert instance._heartbeat_task is None

    asyncio.run(run())


def test_ephemeral_start_skips_periodic_heartbeat():
    """Activity-scoped workers emit only the initial heartbeat — no periodic
    loop, since the pod is short-lived."""
    async def run() -> None:
        instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
        with mock.patch('pool.activities._probe_local_disks',
                        return_value=((0, 0, None), (0, 0, None))):
            await instance.start()
            assert instance._heartbeat_task is None
            await instance.stop()  # idempotent no-op

    asyncio.run(run())


def test_stop_idempotent_when_never_started():
    async def run() -> None:
        instance = _make_instance()
        await instance.stop()  # never started; should not raise

    asyncio.run(run())


def test_heartbeat_long_lived_does_not_report_gpu_slots():
    """Daemon heartbeats do not carry GPU slots; controller owns durable GPU
    capacity from the persisted tier and in-use slots from leases."""
    async def run() -> None:
        instance = _make_instance(PoolStorageMode.DAEMON_ON_PVC)
        with mock.patch('pool.activities._probe_local_disks',
                        return_value=((0, 0, None), (0, 0, None))):
            await instance._send_status_report()
        report = instance._controller.report_status.await_args.args[0]
        assert not hasattr(report.state, 'gpu_slots_total')
        assert not hasattr(report.state, 'gpu_slots_in_use')

    asyncio.run(run())


def test_heartbeat_activity_scoped_does_not_report_gpu_slots():
    """Activity-scoped pods also omit GPU slots; the controller skips
    their inventory entirely (``is_activity_scoped`` branch in
    ``report_status``)."""
    async def run() -> None:
        instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
        with mock.patch('pool.activities._probe_local_disks',
                        return_value=((0, 0, None), (0, 0, None))):
            await instance._send_status_report()
        report = instance._controller.report_status.await_args.args[0]
        assert not hasattr(report.state, 'gpu_slots_total')
        assert not hasattr(report.state, 'gpu_slots_in_use')

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('ssd_device', 'dataplane_device', 'same_filesystem'),
    [(101, 202, False), (101, 101, True)],
)
async def test_dataplane_capacity_status_report(tmp_path, ssd_device,
                                                dataplane_device,
                                                same_filesystem):
    from service_framework._raw_deploy_config import DiskgraphConfig

    data_root = tmp_path / 'ssd'
    dataplane_root = tmp_path / 'dataplane'
    instance = _make_instance(PoolStorageMode.DAEMON_ON_PVC)
    config = SimpleNamespace(
        data_compute=DiskgraphConfig(data_root=str(data_root)),
        data_plane_storage_root=str(dataplane_root),
    )

    with (
            mock.patch('pool.activities.deploy_env', return_value=config),
            mock.patch(
                'pool.activities._probe_local_disks',
                return_value=((10_000, 4_000, ssd_device), (3_000, 2_500,
                                                            dataplane_device)),
            ),
    ):
        await instance._send_status_report()

    state = instance._controller.report_status.await_args.args[0].state
    assert state.ssd_total_bytes == 10_000
    assert state.ssd_used_bytes == 4_000
    assert state.dataplane_total_bytes == 3_000
    assert state.dataplane_used_bytes == 2_500
    assert state.dataplane_free_bytes == 500
    assert state.dataplane_same_filesystem_as_ssd is same_filesystem


def test_combined_disk_probe_contains_unexpected_exception():
    from pool import activities

    with mock.patch('pool.activities._probe_local_disk',
                    side_effect=RuntimeError('unexpected')):
        assert activities._probe_local_disks('/ssd',
                                             '/dataplane') == ((0, 0, None),
                                                               (0, 0, None))


@pytest.mark.asyncio
async def test_status_report_disk_probe_is_bounded_and_single_flight():
    instance = _make_instance()
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_probe(_data_root, _dataplane_root):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5)
        return ((10_000, 4_000, 101), (3_000, 2_500, 202))

    retained_task = None
    try:
        with (
                mock.patch('pool.activities._probe_local_disks',
                           side_effect=blocked_probe),
                mock.patch('pool.activities._DISK_PROBE_WAIT_SECS', 0.01),
        ):
            ticker = asyncio.create_task(asyncio.sleep(0))
            await instance._send_status_report()
            assert ticker.done()
            assert started.is_set()

            state = instance._controller.report_status.await_args.args[0].state
            assert state.ssd_total_bytes == 0
            assert state.dataplane_total_bytes == 0
            assert state.dataplane_same_filesystem_as_ssd is None

            await instance._send_status_report()
            assert calls == 1
            retained_task = instance._disk_probe_task
            assert retained_task is not None
            assert not retained_task.done()
    finally:
        release.set()
        if retained_task is not None:
            await _wait_for_task_done(retained_task)
            await retained_task
            instance._harvest_disk_probe()


@pytest.mark.asyncio
async def test_disk_probe_cache_expiry_and_recovery():
    first = ((10_000, 4_000, 101), (3_000, 2_500, 202))
    second = ((11_000, 4_500, 303), (11_000, 5_000, 303))
    instance = _make_instance()
    now = [100.0]
    instance._monotonic = lambda: now[0]
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    calls = 0

    def probe(_data_root, _dataplane_root):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        refresh_started.set()
        assert release_refresh.wait(timeout=5)
        return second

    trailing_task = None
    try:
        with (
                mock.patch('pool.activities._probe_local_disks',
                           side_effect=probe),
                mock.patch('pool.activities._DISK_PROBE_WAIT_SECS', 0.01),
                mock.patch('pool.activities._DISK_TELEMETRY_MAX_AGE_SECS',
                           60.0),
        ):
            assert await instance._resolve_disk_telemetry('/ssd',
                                                          '/dp') == first
            assert await instance._resolve_disk_telemetry('/ssd',
                                                          '/dp') == first
            await _wait_for_thread_event(refresh_started)

            now[0] += 61.0
            assert await instance._resolve_disk_telemetry(
                '/ssd', '/dp') == ((0, 0, None), (0, 0, None))
            assert calls == 2

            release_refresh.set()
            retained_task = instance._disk_probe_task
            assert retained_task is not None
            await _wait_for_task_done(retained_task)
            await retained_task

            assert await instance._resolve_disk_telemetry('/ssd',
                                                          '/dp') == second
            trailing_task = instance._disk_probe_task
    finally:
        release_refresh.set()
        if trailing_task is not None:
            await _wait_for_task_done(trailing_task)
            await trailing_task
            instance._harvest_disk_probe()


@pytest.mark.asyncio
async def test_disk_probe_task_exception_is_unknown_and_retryable():
    instance = _make_instance()

    with mock.patch('pool.activities._probe_local_disks',
                    side_effect=RuntimeError('unexpected')):
        telemetry = await instance._resolve_disk_telemetry('/ssd', '/dp')

    assert telemetry == ((0, 0, None), (0, 0, None))
    assert instance._disk_probe_task is None


@pytest.mark.asyncio
async def test_cancelled_report_preserves_in_flight_disk_probe():
    instance = _make_instance()
    started = threading.Event()
    release = threading.Event()

    def blocked_probe(_data_root, _dataplane_root):
        started.set()
        assert release.wait(timeout=5)
        return ((10_000, 4_000, 101), (3_000, 2_500, 202))

    retained_task = None
    try:
        with mock.patch('pool.activities._probe_local_disks',
                        side_effect=blocked_probe):
            report = asyncio.create_task(
                instance._resolve_disk_telemetry('/ssd', '/dp'))
            await _wait_for_thread_event(started)
            retained_task = instance._disk_probe_task
            report.cancel()
            with pytest.raises(asyncio.CancelledError):
                await report
            assert instance._disk_probe_task is retained_task
            assert retained_task is not None
            assert not retained_task.cancelled()
    finally:
        release.set()
        if retained_task is not None:
            await _wait_for_task_done(retained_task)
            await retained_task
            instance._harvest_disk_probe()


@pytest.mark.asyncio
async def test_stop_and_restart_do_not_duplicate_blocked_disk_probe():
    instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_probe(_data_root, _dataplane_root):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5)
        return ((10_000, 4_000, 101), (3_000, 2_500, 202))

    retained_task = None
    try:
        with (
                mock.patch('pool.activities._probe_local_disks',
                           side_effect=blocked_probe),
                mock.patch('pool.activities._DISK_PROBE_WAIT_SECS', 0.01),
        ):
            await instance.start()
            assert started.is_set()
            retained_task = instance._disk_probe_task
            await instance.stop()
            assert instance._disk_probe_task is retained_task
            assert retained_task is not None
            assert not retained_task.done()

            await instance.start()
            assert calls == 1
            await instance.stop()
    finally:
        release.set()
        if retained_task is not None:
            await _wait_for_task_done(retained_task)
            await retained_task
            instance._harvest_disk_probe()


@pytest.mark.asyncio
async def test_daemon_disk_checks_pass_blockbuster_raise(tmp_path):
    from async_utils.blockbuster_guard import (
        BlockBusterMode,
        _reset_blockbuster_guard_for_tests,
        activate_blockbuster_guard,
    )
    from kumo.diskgraph import IngestResult
    from service_framework._raw_deploy_config import DiskgraphConfig

    data_root = tmp_path / 'ssd'
    dataplane_root = tmp_path / 'dataplane'
    local_dir = data_root / 'cached-local'
    local_dir.mkdir(parents=True)
    dataplane_root.mkdir()
    (local_dir / 'payload.bin').write_bytes(b'payload')
    config = SimpleNamespace(
        data_compute=DiskgraphConfig(data_root=str(data_root)),
        data_plane_storage_root=str(dataplane_root),
    )
    cached = IngestResult(
        index_path=str(tmp_path / 'idx'),
        col_stats_path=str(tmp_path / 'col'),
        ingest_stats_path=str(tmp_path / 'stats'),
        kumo_stats_dir=str(tmp_path / 'kumo'),
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.has_resumable_ingest.return_value = False
    fake_service.verify_integrity.return_value = cached
    instance = _make_instance()
    stop_ticker = asyncio.Event()

    async def keep_loop_awake():
        while not stop_ticker.is_set():
            await asyncio.sleep(0.001)

    ticker = asyncio.create_task(keep_loop_awake())

    _reset_blockbuster_guard_for_tests()
    activate_blockbuster_guard(mode=BlockBusterMode.RAISE)
    try:
        with mock.patch('pool.activities.deploy_env', return_value=config):
            await instance._send_status_report()

        instance._send_status_report = mock.AsyncMock()
        with (
                mock.patch('pool.activities.get_service',
                           return_value=fake_service),
                mock.patch('pool.activities.get_persist_dir',
                           return_value=str(dataplane_root)),
        ):
            result = await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)
            assert result is cached
    finally:
        _reset_blockbuster_guard_for_tests()
        stop_ticker.set()
        await ticker


def test_dataplane_capacity_probe_object_store_is_unknown():
    from pool import activities

    assert activities._probe_local_disk('s3://bucket/dataplane') == (0, 0, None)


def test_dataplane_capacity_probe_missing_leaf_uses_existing_ancestor(tmp_path):
    from pool import activities

    total, used, device = activities._probe_local_disk(
        str(tmp_path / 'missing' / 'leaf'))

    assert total > 0
    assert 0 <= used <= total
    assert device == tmp_path.stat().st_dev


def test_dataplane_capacity_probe_preserves_writer_available_bytes(tmp_path):
    from pool import activities

    usage = SimpleNamespace(total=1_000, used=600, free=300)
    with mock.patch('pool.activities.shutil.disk_usage', return_value=usage):
        total, used, _ = activities._probe_local_disk(str(tmp_path))

    assert total - used == usage.free


def test_dataplane_capacity_probe_oserror_is_unknown(tmp_path):
    from pool import activities

    with mock.patch('pool.activities.shutil.disk_usage',
                    side_effect=OSError('unavailable')):
        assert activities._probe_local_disk(str(tmp_path)) == (0, 0, None)


@pytest.mark.asyncio
async def test_status_reports_are_serialized_with_ready_snapshot_capture():
    instance = _make_instance()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    reports = []

    async def report_status(report):
        reports.append(report)
        if len(reports) == 1:
            first_started.set()
            await release_first.wait()

    instance._controller.report_status.side_effect = report_status
    instance._ready_snapshots.add('snap-1')
    with mock.patch('pool.activities._probe_local_disks',
                    return_value=((0, 0, None), (0, 0, None))):
        heartbeat = asyncio.create_task(instance._send_status_report())
        await first_started.wait()

        instance._ready_snapshots.discard('snap-1')
        corruption = asyncio.create_task(
            instance._send_status_report(
                WorkerEvent(type=WorkerEventType.SSD_CACHE_CORRUPTED,
                            snapshot_id='snap-1')))
        await asyncio.sleep(0)
        assert len(reports) == 1

        release_first.set()
        await asyncio.gather(heartbeat, corruption)
    assert reports[0].state.ready_snapshots == ['snap-1']
    assert reports[1].state.ready_snapshots == []
    assert reports[1].event.type == WorkerEventType.SSD_CACHE_CORRUPTED


# ----------------------------------------------------------------------
# _activity_scope: the activity-scoped retry contract.
#
# Each test exercises one corner of the contract that worker_main.py
# relies on in its activity-scoped failure-exit branch. The pod's exit code drives
# K8s Job behavior:
#
#   * exit 0 → Job complete, pod reclaimed.
#   * exit 1 → K8s ``restartPolicy: OnFailure`` restarts the pod;
#              Temporal reschedules the retry attempt onto the new pod.
#
# A spurious exit 1 (e.g., on workflow cancellation) burns a retry
# attempt for nothing and has the new pod poll for a task that will
# never arrive.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activity_scope_success_does_not_set_failed_flag():
    instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
    async with instance._activity_scope():
        pass  # successful activity body
    assert instance._activity_failed is False
    assert instance.activity_done.is_set() is True


@pytest.mark.asyncio
async def test_activity_scope_failure_sets_failed_flag():
    """Retryable activity failure must set ``_activity_failed`` so
    worker_main exits 1, K8s restarts the pod, and Temporal reschedules
    the retry attempt onto the fresh pod."""
    instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
    with pytest.raises(RuntimeError, match='boom'):
        async with instance._activity_scope():
            raise RuntimeError('boom')
    assert instance._activity_failed is True
    assert instance.activity_done.is_set() is True


@pytest.mark.asyncio
async def test_activity_scope_cancellation_sets_failed_flag():
    """Cancellation is treated as a retryable failure on purpose. The
    primary case is heartbeat-timeout cancellation: Temporal already
    scheduled a retry attempt, and the new attempt needs a worker
    (i.e., a fresh pod) to land on. Workflow cancellation also lands
    here and the pod restart is dangling — the controller-side
    workflow-cancellation path is responsible for deleting the Job
    in that case."""
    instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
    with pytest.raises(asyncio.CancelledError):
        async with instance._activity_scope():
            raise asyncio.CancelledError()
    assert instance._activity_failed is True
    assert instance.activity_done.is_set() is True


@pytest.mark.asyncio
async def test_long_lived_activity_scope_does_not_fire_self_exit():
    """DAEMON_ON_PVC workers stay alive across many activities — failure
    flag is still tracked, but ``activity_done`` (the self-exit signal)
    must NOT fire since the worker keeps polling."""
    instance = _make_instance(PoolStorageMode.DAEMON_ON_PVC)
    with pytest.raises(RuntimeError):
        async with instance._activity_scope():
            raise RuntimeError('boom')
    assert instance._activity_failed is True
    assert instance.activity_done.is_set() is False


@pytest.mark.parametrize('exc_cls', [ValueError, KeyError, TypeError])
@pytest.mark.asyncio
async def test_activity_scope_non_retryable_does_not_set_failed_flag(exc_cls):
    """Non-retryable errors (ValueError / KeyError / TypeError /
    ApplicationError(non_retryable=True) / etc.) terminally fail the
    activity — Temporal will NOT schedule a retry attempt. Pod must
    exit 0 to avoid K8s OnFailure restart that would have the new pod
    poll for a task that never arrives."""
    instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
    with pytest.raises(exc_cls):
        async with instance._activity_scope():
            raise exc_cls('non-retryable')
    assert instance._activity_failed is False
    assert instance.activity_done.is_set() is True


@pytest.mark.asyncio
async def test_activity_scope_application_error_non_retryable():
    """``ApplicationError(non_retryable=True)`` is the canonical
    "fail this activity, don't retry" signal — used by activity bodies
    to short-circuit Temporal's retry policy. Same exit-0 behavior."""
    from temporalio.exceptions import ApplicationError
    instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
    with pytest.raises(ApplicationError):
        async with instance._activity_scope():
            raise ApplicationError('terminal', non_retryable=True)
    assert instance._activity_failed is False
    assert instance.activity_done.is_set() is True


@pytest.mark.asyncio
async def test_activity_scope_application_error_retryable_sets_failed_flag():
    """``ApplicationError(non_retryable=False)`` is retryable — Temporal
    will schedule another attempt, so the pod must exit 1 so K8s
    restarts and the retry lands on the fresh pod."""
    from temporalio.exceptions import ApplicationError
    instance = _make_instance(PoolStorageMode.ACTIVITY_ON_PVC)
    with pytest.raises(ApplicationError):
        async with instance._activity_scope():
            raise ApplicationError('transient', non_retryable=False)
    assert instance._activity_failed is True
    assert instance.activity_done.is_set() is True


def test_module_level_wrappers_are_activities():
    """The workflow-side wrappers must be Temporal-registered so they
    can be passed to ``Worker(activities=[...])`` on the workflow
    worker. ``activity._Definition.must_from_callable`` raises if the
    decorator wasn't applied."""
    activity._Definition.must_from_callable(place_snapshot_activity)
    activity._Definition.must_from_callable(release_snapshot_activity)


# ----------------------------------------------------------------------
# RpcError → ApplicationError translation
#
# The AutoRPC client raises ``RpcError(non_retryable=...)`` based on
# whether the controller-side failure was deterministic. Temporal's
# retry policy keys off type names, and ``RpcError`` isn't in the
# default non-retryable list — so without this wrapper translation,
# deterministic controller errors (RBAC denial, missing IRSA, invalid
# spec) would retry for the full ``schedule_to_close`` budget. The
# wrappers re-raise as ``ApplicationError`` carrying the original
# type / message / non_retryable bit, which Temporal honors.
# ----------------------------------------------------------------------


def test_place_retry_policy_is_bounded_by_schedule_to_close():
    """Transient retries are unbounded only inside the 45-minute budget."""
    from pool import timeouts
    from pool.wf_coordinator import _PLACE_RETRY_POLICY

    assert _PLACE_RETRY_POLICY.maximum_attempts == 0
    assert timeouts.PLACE_SCHEDULE_TO_CLOSE.total_seconds() == 45 * 60


@pytest.mark.asyncio
@pytest.mark.parametrize('non_retryable', [True, False])
async def test_place_snapshot_activity_translates_rpc_error(non_retryable):
    from temporalio.exceptions import ApplicationError

    from pool.dataclasses import (
        PlacementRequest,
        ResourceHint,
        SnapshotActivity,
    )
    from temporalio_commons.auto_rpc_exception import RpcError

    request = PlacementRequest(
        snapshot_id='NA-deadbeefdeadbeefdeadbeefdeadbeef@v1',
        job_id='job-1',
        workflow_id='wf-1',
        activity=SnapshotActivity.GEN_TT,
        resource_hint=ResourceHint(),
    )
    rpc_err = RpcError(
        'PoolController',
        'place_snapshot',
        error_type='K8sPermissionError',
        message='pods is forbidden',
        stack_trace='',
        non_retryable=non_retryable,
    )
    fake_client = _MockController()
    fake_client.place_snapshot.side_effect = rpc_err

    with mock.patch('pool.activities.get_controller_client',
                    return_value=fake_client), \
            mock.patch('pool.activities.deploy_env'):
        with pytest.raises(ApplicationError) as excinfo:
            await place_snapshot_activity(request)

    assert excinfo.value.non_retryable is non_retryable
    assert excinfo.value.type == 'K8sPermissionError'
    assert 'pods is forbidden' in excinfo.value.message
    assert excinfo.value.__cause__ is rpc_err


@pytest.mark.asyncio
@pytest.mark.parametrize('non_retryable', [True, False])
async def test_release_snapshot_activity_translates_rpc_error(non_retryable):
    from temporalio.exceptions import ApplicationError

    from temporalio_commons.auto_rpc_exception import RpcError

    rpc_err = RpcError(
        'PoolController',
        'release_lease',
        error_type='PoolWorkerNotFoundError',
        message='no such uid',
        stack_trace='',
        non_retryable=non_retryable,
    )
    fake_client = _MockController()
    fake_client.release_lease.side_effect = rpc_err

    with mock.patch('pool.activities.get_controller_client',
                    return_value=fake_client), \
            mock.patch('pool.activities.deploy_env'):
        with pytest.raises(ApplicationError) as excinfo:
            await release_snapshot_activity(
                'NA-deadbeefdeadbeefdeadbeefdeadbeef@v1', 'job-1')

    assert excinfo.value.non_retryable is non_retryable
    assert excinfo.value.type == 'PoolWorkerNotFoundError'


@pytest.mark.asyncio
@pytest.mark.parametrize((
    'error_type', 'expected_non_retryable', 'expected_public_message'
), [
    (
        'ProvisionTimeoutError',
        False,
        'Your job could not acquire a GPU/compute worker within the '
        'allowed time. GPU capacity may be temporarily limited. '
        'Please retry; if the problem persists, contact Kumo support.',
    ),
    (
        'ProvisionTerminalError',
        False,
        'Your job could not start because the compute pod failed to '
        'initialize (e.g. bad image tag, missing secret, or invalid '
        'container configuration). This is an infrastructure issue — '
        'please contact Kumo support.',
    ),
    (
        'InsufficientDataplaneCapacityError',
        True,
        'Your graph cannot be ingested because the local dataplane volume '
        'does not have enough available space. Free or expand that volume, '
        'move the dataplane root to supported object storage, or reduce the '
        'dataset size.',
    ),
    (
        'ExistingWorkerUnavailableError',
        True,
        ('The DiskGraph worker reserved for your job is no longer '
         'available. This is an infrastructure issue, not a problem with '
         'your graph or model. Please retry; if the problem persists, '
         'contact Kumo support.'),
    ),
])
async def test_place_snapshot_activity_includes_public_exception_for_placement_errors(
        error_type, expected_non_retryable, expected_public_message):
    """Pool placement errors include a PublicException in ApplicationError.details.

    This ensures that the normal public-exception extraction path can surface
    the allowlisted placement failure's specific remediation instead of a
    generic fallback, whether the error is retried or terminal.
    """
    from temporalio.exceptions import ApplicationError

    from common_utils.error import PublicException
    from pool.dataclasses import (
        PlacementRequest,
        ResourceHint,
        SnapshotActivity,
    )
    from temporalio_commons.auto_rpc_exception import RpcError

    rpc_err = RpcError(
        'PoolController',
        'place_snapshot',
        error_type=error_type,
        message='injected error',
        stack_trace='',
        non_retryable=expected_non_retryable,
    )
    request = PlacementRequest(
        snapshot_id='NA-deadbeefdeadbeefdeadbeefdeadbeef@v1',
        job_id='job-1',
        workflow_id='wf-1',
        activity=SnapshotActivity.GEN_TT,
        resource_hint=ResourceHint(),
    )
    fake_client = _MockController()
    fake_client.place_snapshot.side_effect = rpc_err

    with mock.patch('pool.activities.get_controller_client',
                    return_value=fake_client), \
            mock.patch('pool.activities.deploy_env'):
        with pytest.raises(ApplicationError) as excinfo:
            await place_snapshot_activity(request)

    app_err = excinfo.value
    assert app_err.type == error_type
    assert app_err.non_retryable is expected_non_retryable
    # Must include a PublicException detail so timeout mapper picks it up.
    assert app_err.details, 'Expected PublicException in ApplicationError.details'
    public_ex = PublicException.from_dict(app_err.details[0])
    assert public_ex is not None
    assert public_ex.message == expected_public_message


def test_class_methods_are_activities():
    """Every public coroutine on ``SnapshotActivities`` that's intended
    to be a Temporal activity has the marker."""
    instance = _make_instance()
    for name in [
            'ensure_snapshot',
            'delete_snapshot',
            'gen_tt',
            'build_ttng_custom_pred_table_bundle',
            'build_ttng_custom_train_table_bundle',
            'materialize_training_task',
            'materialize_prediction_task',
            'train',
            'predict',
    ]:
        method = getattr(instance, name)
        # Attribute fallthrough on bound methods reaches the underlying
        # decorated function.
        activity._Definition.must_from_callable(method)


@pytest.mark.asyncio
async def test_build_ttng_custom_train_table_bundle_ensures_snapshot_first():
    """The pooled custom-TT gen wrapper must ``_ensure_snapshot`` the parent
    graph index onto local SSD BEFORE delegating to the gen body — otherwise
    the wheel gen can't open ``index_metadata.json`` on the [s3] lane
    (kumo#29738 residency fix)."""
    instance = _make_instance()
    order: list[str] = []

    async def _fake_ensure_snapshot(snapshot_id, *_a, **_k):
        order.append(f'ensure:{snapshot_id}')

    delegated: dict = {}

    async def _fake_run_diskgraph_async(fn, *args, **kwargs):
        order.append('delegate')
        delegated['fn'] = fn
        delegated['args'] = args

    with mock.patch.object(instance, '_ensure_snapshot',
                           _fake_ensure_snapshot), \
            mock.patch('pool.activities._run_diskgraph_async',
                       _fake_run_diskgraph_async):
        await instance.build_ttng_custom_train_table_bundle(
            'graph-11111111111111111111111111111111@v1',
            'job-input',
            'settings',
            's3://bucket/custom_train_table',
            'weight',
        )

    # ensure_snapshot ran before the gen body was invoked.
    assert order[0].startswith('ensure:')
    assert order[1] == 'delegate'
    # Delegated to the real (undecorated) gen body with the job args, minus the
    # snapshot_id (which is pool-routing metadata, not a gen input).
    from rest.backends.ml_backend.activities.materialize_training_task import \
        build_ttng_custom_train_table_bundle as _gen_body
    assert delegated['fn'] is _gen_body
    assert delegated['args'] == ('job-input', 'settings',
                                 's3://bucket/custom_train_table', 'weight')

    order.clear()
    with mock.patch.object(instance, '_ensure_snapshot',
                           _fake_ensure_snapshot), \
            mock.patch('pool.activities._run_diskgraph_async',
                       _fake_run_diskgraph_async):
        await instance.build_ttng_custom_pred_table_bundle(
            'graph-11111111111111111111111111111111@v1',
            'job-input',
            'settings',
            'abfss://filesystem@account.dfs.core.windows.net/custom_pred.parquet',
        )
    from rest.backends.ml_backend.activities.materialize_prediction_task import \
        build_ttng_custom_pred_table_bundle as _pred_body
    assert order == [
        'ensure:graph-11111111111111111111111111111111@v1', 'delegate'
    ]
    assert delegated['fn'] is _pred_body
    assert delegated['args'] == (
        'job-input', 'settings',
        'abfss://filesystem@account.dfs.core.windows.net/custom_pred.parquet')


# ----------------------------------------------------------------------
# Subprocess cancellation: workflow cancel → SIGTERM → SIGKILL fallback
# ----------------------------------------------------------------------


class _FakeProc:
    """asyncio.subprocess.Process stand-in.

    Captures terminate/kill calls and exposes a controllable wait()
    coroutine so tests can simulate "subprocess never exits" /
    "subprocess exits after SIGTERM" / "subprocess only dies on
    SIGKILL".
    """
    def __init__(self, *, exit_after_term: bool, exit_after_kill: bool = True):
        self.pid = 12345
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self._exit_after_term = exit_after_term
        self._exit_after_kill = exit_after_kill
        self._exit_event = asyncio.Event()
        # An empty-payload stdout that closes immediately so the drain
        # task in ``_run_kumo_subprocess`` exits without blocking.
        self.stdout = _ClosedStreamReader()

    def terminate(self):
        self.terminate_called = True
        if self._exit_after_term:
            self.returncode = -15  # SIGTERM
            self._exit_event.set()

    def kill(self):
        self.kill_called = True
        if self._exit_after_kill:
            self.returncode = -9  # SIGKILL
            self._exit_event.set()

    async def wait(self):
        await self._exit_event.wait()
        return self.returncode


class _ClosedStreamReader:
    """Minimal asyncio.StreamReader stand-in whose read returns
    EOF (b'') immediately so the drain task short-circuits."""
    async def read(self, _size: int) -> bytes:
        return b''


class _FailedStreamReader:
    async def read(self, _size: int) -> bytes:
        raise RuntimeError('stdout read failed')


@pytest.mark.asyncio
async def test_subprocess_output_drain_handles_long_tqdm_carriage_returns(
        caplog):
    reader = asyncio.StreamReader(limit=64)
    updates = [f'Predicting {index}/5000'.encode() for index in range(5000)]
    reader.feed_data(b'\r'.join(updates) + b'\rprediction complete\n'
                     b'unterminated final record')
    reader.feed_eof()

    with caplog.at_level('INFO'):
        await _drain_subprocess_output(reader, 'kumo-pred', 'job-1')

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.endswith('Predicting 0/5000') for message in messages)
    assert any(message.endswith('Predicting 4999/5000') for message in messages)
    assert sum('Predicting ' in message for message in messages) == 2
    assert any(message.endswith('prediction complete') for message in messages)
    assert any(
        message.endswith('unterminated final record') for message in messages)


@pytest.mark.asyncio
async def test_subprocess_output_drain_bounds_undelimited_records():
    payload = b'x' * (_SUBPROCESS_OUTPUT_MAX_RECORD_BYTES * 3 + 17)
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()

    with mock.patch('pool.activities.log.info') as info:
        await _drain_subprocess_output(reader, 'kumo-pred', 'job-1')

    records = [call.args[3] for call in info.call_args_list]
    assert b''.join(record.encode() for record in records) == payload
    assert max(map(len, records)) <= _SUBPROCESS_OUTPUT_MAX_RECORD_BYTES


@pytest.mark.asyncio
async def test_subprocess_output_drain_failure_terminates_child(monkeypatch):
    fake_proc = _FakeProc(exit_after_term=True)
    fake_proc.stdout = _FailedStreamReader()

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        return fake_proc

    monkeypatch.setattr('asyncio.create_subprocess_exec',
                        _fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match='stdout read failed'):
        await _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json',
                                   mock.Mock())

    assert fake_proc.terminate_called
    assert not fake_proc.kill_called


@pytest.mark.parametrize(
    ('iterations_remaining', 'refit_trainval', 'refit_full', 'stage_name'),
    [
        (1, False, False, 'Training'),
        (0, True, False, 'Refitting on Validation Data'),
        (0, False, True, 'Refitting on Full Data'),
        (0, False, False, 'Training'),
    ],
)
def test_training_progress_output_selects_stage(iterations_remaining,
                                                refit_trainval, refit_full,
                                                stage_name):
    output = training_progress_output(
        TrainingJobProgress(iterations_remaining=iterations_remaining),
        SimpleNamespace(
            refit_trainval=refit_trainval,
            refit_full=refit_full,
        ),
    )

    assert output is not None
    assert output.status == 'RUNNING'
    assert output.log.stage_name == stage_name
    assert output.progress.iterations_remaining == iterations_remaining


def test_training_progress_output_skips_missing_progress():
    output = training_progress_output(
        None, SimpleNamespace(refit_trainval=False, refit_full=False))

    assert output is None


@pytest.mark.asyncio
async def test_kumo_pred_subprocess_does_not_publish_training_progress(
        monkeypatch):
    fake_proc = _FakeProc(exit_after_term=False)
    subprocess_kwargs = {}

    async def _fake_create_subprocess_exec(*_args, **kwargs):
        subprocess_kwargs.update(kwargs)
        return fake_proc

    publish = mock.AsyncMock()
    heartbeat = mock.Mock()

    def _finish_on_heartbeat():
        fake_proc.returncode = 0
        fake_proc._exit_event.set()

    heartbeat.side_effect = _finish_on_heartbeat
    monkeypatch.setattr('asyncio.create_subprocess_exec',
                        _fake_create_subprocess_exec)
    monkeypatch.setattr('pool.activities._SUBPROCESS_HEARTBEAT_INTERVAL_SECS',
                        0)
    monkeypatch.setattr('pool.activities.activity.heartbeat', heartbeat)
    monkeypatch.setattr('pool.activities.publish_training_progress', publish)
    # yapf's auto-wrap for this line produces flake8 E251 (mismatched spaces
    # around `data_root=` at the wrap point); pin the wrapping by hand.
    # yapf: disable
    monkeypatch.setattr(
        'pool.activities.deploy_env',
        lambda: SimpleNamespace(
            data_compute=SimpleNamespace(data_root='/mnt/localssd')),
    )
    # yapf: enable
    monkeypatch.delenv('KUMO_BP_OUTPUT_SPOOL_ROOT', raising=False)

    result = await _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json',
                                        mock.Mock())

    assert result == 'SUCCEEDED'
    assert subprocess_kwargs['start_new_session'] is True
    heartbeat.assert_called_once()
    assert subprocess_kwargs['env']['KUMO_BP_OUTPUT_SPOOL_ROOT'] == (
        '/mnt/localssd/bp-output-spool')
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_kumo_pred_subprocess_preserves_explicit_spool_root(monkeypatch):
    fake_proc = _FakeProc(exit_after_term=False)
    fake_proc.returncode = 0
    fake_proc._exit_event.set()
    subprocess_kwargs = {}

    async def _fake_create_subprocess_exec(*_args, **kwargs):
        subprocess_kwargs.update(kwargs)
        return fake_proc

    monkeypatch.setattr('asyncio.create_subprocess_exec',
                        _fake_create_subprocess_exec)
    monkeypatch.setenv('KUMO_BP_OUTPUT_SPOOL_ROOT', '/explicit/spool')

    result = await _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json',
                                        None)

    assert result == 'SUCCEEDED'
    assert subprocess_kwargs['env']['KUMO_BP_OUTPUT_SPOOL_ROOT'] == (
        '/explicit/spool')


@pytest.mark.asyncio
async def test_subprocess_nonzero_exit_returns_failed(monkeypatch):
    fake_proc = _FakeProc(exit_after_term=False)
    fake_proc.returncode = 7
    fake_proc._exit_event.set()

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        return fake_proc

    monkeypatch.setattr('asyncio.create_subprocess_exec',
                        _fake_create_subprocess_exec)

    result = await _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json',
                                        None)

    assert result == 'FAILED'


@pytest.mark.asyncio
async def test_kumo_auto_progress_publish_timeout_does_not_block_subprocess(
        monkeypatch):
    fake_proc = _FakeProc(exit_after_term=False)

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        return fake_proc

    async def _slow_publish(*_args, **_kwargs):
        await asyncio.Event().wait()

    real_wait_for = asyncio.wait_for
    publish_timeout_seen = False

    async def _fake_wait_for(coro, timeout):
        nonlocal publish_timeout_seen
        if (timeout == TRAINING_PROGRESS_PUBLISH_TIMEOUT_SECS
                and not publish_timeout_seen):
            publish_timeout_seen = True
            coro.close()
            fake_proc.returncode = 0
            fake_proc._exit_event.set()
            raise asyncio.TimeoutError()
        return await real_wait_for(coro, timeout)

    heartbeat = mock.Mock()
    monkeypatch.setattr("asyncio.create_subprocess_exec",
                        _fake_create_subprocess_exec)
    monkeypatch.setattr("asyncio.wait_for", _fake_wait_for)
    monkeypatch.setattr("pool.activities._SUBPROCESS_HEARTBEAT_INTERVAL_SECS",
                        0)
    monkeypatch.setattr("pool.activities.activity.heartbeat", heartbeat)
    monkeypatch.setattr("pool.activities.publish_training_progress",
                        _slow_publish)

    result = await _run_kumo_subprocess("kumo-auto", "job-1", "/tmp/cfg.json",
                                        mock.Mock())

    assert result == "SUCCEEDED"
    assert publish_timeout_seen is True
    heartbeat.assert_called_once()


def test_job_output_store_is_self_contained_and_cached():
    """The progress store is created on first use, in self-contained mode,
    and cached so repeated calls return the same instance."""
    instance = _make_instance()

    # Not created until something actually asks for it.
    assert instance._job_output_store is None

    store = instance._get_job_output_store()
    assert isinstance(store, MLJobOutputStore)
    assert store._self_contained is True
    # Cached: same instance on subsequent calls.
    assert instance._get_job_output_store() is store


@pytest.mark.asyncio
async def test_job_output_store_not_created_without_train():
    """A pure-ingest daemon never runs ``train``; constructing and stopping
    the worker must never create the store (no StateDB dependency)."""
    instance = _make_instance()

    await instance.stop()

    assert instance._job_output_store is None


@pytest.mark.asyncio
async def test_subprocess_cancellation_sends_sigterm():
    """Workflow cancellation must propagate to the subprocess.

    Without the cancellation handling in ``_run_kumo_subprocess``, the
    activity is cancelled but kumo-auto / kumo-pred continue to run
    (training can take hours), holding GPU + SSD until natural exit.
    """
    fake_proc = _FakeProc(exit_after_term=True)

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        return fake_proc

    async def _runner():
        with mock.patch('asyncio.create_subprocess_exec',
                        _fake_create_subprocess_exec):
            await _run_kumo_subprocess('kumo-auto', 'job-1', '/tmp/cfg.json',
                                       mock.Mock())

    task = asyncio.create_task(_runner())
    # Let _run_kumo_subprocess get past the create_subprocess_exec call
    # and into its heartbeat loop awaiting proc.wait().
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_proc.terminate_called, (
        'workflow cancel should SIGTERM the subprocess')
    assert not fake_proc.kill_called, (
        'subprocess exited cleanly on SIGTERM; SIGKILL must not fire')


@pytest.mark.asyncio
async def test_subprocess_cancellation_falls_back_to_sigkill():
    """If the subprocess ignores SIGTERM (no exit within 10s), the
    cancellation handler must follow up with SIGKILL so the activity
    doesn't hang indefinitely on cleanup."""
    # exit_after_term=False simulates a subprocess that ignores SIGTERM.
    fake_proc = _FakeProc(exit_after_term=False, exit_after_kill=True)

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        return fake_proc

    real_wait_for = asyncio.wait_for

    async def _fake_wait_for(coro, timeout):
        # Fast-forward the 10s post-SIGTERM grace window to make the
        # test instant. Real ``asyncio.wait_for`` semantics: cancel the
        # awaitable + raise TimeoutError.
        if timeout == 10:
            task = asyncio.ensure_future(coro)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # expected after task.cancel()
            raise asyncio.TimeoutError()
        return await real_wait_for(coro, timeout)

    async def _runner():
        with mock.patch('asyncio.create_subprocess_exec',
                        _fake_create_subprocess_exec):
            with mock.patch('asyncio.wait_for', _fake_wait_for):
                await _run_kumo_subprocess('kumo-auto', 'job-1',
                                           '/tmp/cfg.json', mock.Mock())

    task = asyncio.create_task(_runner())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_proc.terminate_called
    assert fake_proc.kill_called, (
        'SIGTERM ignored; SIGKILL must fire as the fallback')


@pytest.mark.asyncio
async def test_repeated_cancellation_still_kills_subprocess():
    """Cleanup cannot leak a child if its graceful wait is also cancelled."""
    fake_proc = _FakeProc(exit_after_term=False, exit_after_kill=True)

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        return fake_proc

    real_wait_for = asyncio.wait_for

    async def _fake_wait_for(coro, timeout):
        if timeout == 10:
            raise asyncio.CancelledError()
        return await real_wait_for(coro, timeout)

    async def _runner():
        with mock.patch('asyncio.create_subprocess_exec',
                        _fake_create_subprocess_exec):
            with mock.patch('asyncio.wait_for', _fake_wait_for):
                await _run_kumo_subprocess('kumo-auto', 'job-1',
                                           '/tmp/cfg.json', mock.Mock())

    task = asyncio.create_task(_runner())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_proc.terminate_called
    assert fake_proc.kill_called


@pytest.mark.parametrize('repeat_cancel', [False, True])
@pytest.mark.parametrize('supervisor_available', [True, False])
@pytest.mark.asyncio
async def test_subprocess_cancellation_reaps_native_descendant_session(
        tmp_path, monkeypatch, repeat_cancel, supervisor_available):
    """Strong and degraded cancellation own the native process tree."""
    if (supervisor_available
            and not _subprocess_supervisor_capabilities_available()):
        pytest.skip('strong pidfd/procfs supervisor unavailable')
    child_pid_path = tmp_path / 'native-child.pid'
    parent_script = tmp_path / 'kumo-pred'
    parent_script.write_text(f'''\
#!{sys.executable}
import signal
import subprocess
import sys

child = subprocess.Popen([
    sys.executable,
    '-c',
    'import signal,time; '
    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
    'print("ready", flush=True); time.sleep(60)',
], stdout=subprocess.PIPE, text=True)
assert child.stdout.readline().strip() == 'ready'
import os as _os
with open({str(child_pid_path)!r} + '.tmp', 'w') as pid_file:
    pid_file.write(str(child.pid))
_os.replace({str(child_pid_path)!r} + '.tmp', {str(child_pid_path)!r})
child.wait()
''')
    parent_script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    real_exists = os.path.exists
    monkeypatch.setattr(
        'pool.activities.osp.exists', lambda path:
        (False if str(path).endswith('/bin/kumo-pred') else real_exists(path)))
    monkeypatch.setattr('pool.activities._SUBPROCESS_TERM_GRACE_SECS', 0.2)
    monkeypatch.setattr('pool.activities._SUBPROCESS_KILL_GRACE_SECS', 1.0)
    monkeypatch.setattr(
        'pool.activities._subprocess_supervisor_capabilities_available',
        lambda: supervisor_available,
    )

    task = asyncio.create_task(
        _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json', None))
    for _ in range(500):
        if child_pid_path.exists():
            break
        if task.done():
            _ = await task
        await asyncio.sleep(0.01)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text())

    task.cancel()
    if repeat_cancel:
        await asyncio.sleep(0.05)
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)

    try:
        process_stat = Path(f'/proc/{child_pid}/stat').read_bytes()
    except FileNotFoundError:
        child_state = None
    else:
        child_state = process_stat[process_stat.rfind(b')') + 2:].split()[0]
    if supervisor_available:
        assert child_state is None
    else:
        assert child_state in (None, b'Z')


@pytest.mark.asyncio
async def test_subprocess_capture_failure_never_starts_real_command(
        tmp_path, monkeypatch):
    marker = tmp_path / 'command-started'
    command = tmp_path / 'kumo-pred'
    command.write_text(f'''\
#!{sys.executable}
from pathlib import Path
Path({str(marker)!r}).write_text('started')
''')
    command.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    real_exists = os.path.exists
    monkeypatch.setattr(
        'pool.activities.osp.exists', lambda path:
        (False if str(path).endswith('/bin/kumo-pred') else real_exists(path)))
    monkeypatch.setattr(
        'pool.activities._subprocess_supervisor_capabilities_available',
        lambda: True)
    monkeypatch.setattr(
        'pool.activities._OwnedSubprocessSession.capture',
        mock.Mock(side_effect=RuntimeError('synthetic capture failure')))

    with pytest.raises(RuntimeError, match='synthetic capture failure'):
        await _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json', None)

    assert not marker.exists()


@pytest.mark.parametrize('supervisor_available', [True, False])
@pytest.mark.asyncio
async def test_subprocess_success_waits_for_orphan_reap(tmp_path, monkeypatch,
                                                        supervisor_available):
    """A parent rc=0 cleans its native child before reporting success."""
    if (supervisor_available
            and not _subprocess_supervisor_capabilities_available()):
        pytest.skip('strong pidfd/procfs supervisor unavailable')
    child_pid_path = tmp_path / 'native-child.pid'
    command = tmp_path / 'kumo-pred'
    command.write_text(f'''\
#!{sys.executable}
import subprocess
import sys

child = subprocess.Popen([
    sys.executable,
    '-c',
    'import signal,time; '
    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
    'print("ready", flush=True); time.sleep(60)',
], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
assert child.stdout.readline().strip() == 'ready'
child.stdout.close()
import os as _os
with open({str(child_pid_path)!r} + '.tmp', 'w') as pid_file:
    pid_file.write(str(child.pid))
_os.replace({str(child_pid_path)!r} + '.tmp', {str(child_pid_path)!r})
# Exit zero without waiting; the supervisor must adopt, terminate, and reap.
''')
    command.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    real_exists = os.path.exists
    monkeypatch.setattr(
        'pool.activities.osp.exists', lambda path:
        (False if str(path).endswith('/bin/kumo-pred') else real_exists(path)))
    monkeypatch.setattr('pool.activities._SUBPROCESS_TERM_GRACE_SECS', 0.2)
    monkeypatch.setattr('pool.activities._SUBPROCESS_KILL_GRACE_SECS', 1.0)
    monkeypatch.setattr(
        'pool.activities._subprocess_supervisor_capabilities_available',
        lambda: supervisor_available,
    )

    result = await asyncio.wait_for(
        _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json', None),
        timeout=3)

    assert result == 'SUCCEEDED'
    child_pid = int(child_pid_path.read_text())
    try:
        process_stat = Path(f'/proc/{child_pid}/stat').read_bytes()
    except FileNotFoundError:
        child_state = None
    else:
        child_state = process_stat[process_stat.rfind(b')') + 2:].split()[0]
    if supervisor_available:
        assert child_state is None
    else:
        # The degraded fallback delivers SIGKILL, but without subreaper
        # support the container init may not have reaped the orphan yet.
        assert child_state in (None, b'Z')


@pytest.mark.asyncio
async def test_owned_subprocess_preserves_nonzero_exit(tmp_path, monkeypatch):
    command = tmp_path / 'kumo-pred'
    command.write_text(f'''\
#!{sys.executable}
raise SystemExit(7)
''')
    command.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    real_exists = os.path.exists
    monkeypatch.setattr(
        'pool.activities.osp.exists', lambda path:
        (False if str(path).endswith('/bin/kumo-pred') else real_exists(path)))

    result = await _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json',
                                        None)

    assert result == 'FAILED'


@pytest.mark.skipif(os.getpid() != 1,
                    reason='requires production-like Python PID 1 adoption')
@pytest.mark.asyncio
async def test_supervisor_early_exit_reclaims_adopted_descendants(
        tmp_path, monkeypatch):
    process_path = tmp_path / 'owned-processes'
    command = tmp_path / 'kumo-pred'
    command.write_text(f'''\
#!{sys.executable}
import os
import subprocess
import sys
import time
from pathlib import Path

devnull = os.open('/dev/null', os.O_RDWR)
os.dup2(devnull, 1)
os.dup2(devnull, 2)
os.close(devnull)
child = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
import os as _os
Path({str(process_path)!r} + '.tmp').write_text(
    f'{{os.getppid()}} {{os.getpid()}} {{child.pid}}')
_os.replace({str(process_path)!r} + '.tmp', {str(process_path)!r})
time.sleep(60)
''')
    command.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    real_exists = os.path.exists
    monkeypatch.setattr(
        'pool.activities.osp.exists', lambda path:
        (False if str(path).endswith('/bin/kumo-pred') else real_exists(path)))
    monkeypatch.setattr('pool.activities._SUBPROCESS_TERM_GRACE_SECS', 0.2)
    monkeypatch.setattr('pool.activities._SUBPROCESS_KILL_GRACE_SECS', 1.0)

    task = asyncio.create_task(
        _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json', None))
    for _ in range(500):
        if process_path.exists():
            break
        if task.done():
            _ = await task
        await asyncio.sleep(0.01)
    assert process_path.exists()
    supervisor_pid, command_pid, child_pid = map(
        int,
        process_path.read_text().split())

    os.kill(supervisor_pid, signal.SIGKILL)
    with pytest.raises(RuntimeError, match='including zombies'):
        await asyncio.wait_for(task, timeout=5)

    assert not Path(f'/proc/{supervisor_pid}').exists()
    assert not Path(f'/proc/{command_pid}').exists()
    assert not Path(f'/proc/{child_pid}').exists()


def test_process_identity_includes_proc_start_time():
    identity = _read_process_identity(os.getpid())

    assert identity.start_time_ticks > 0
    assert _OwnedSubprocessSession._same_stable_identity(identity, identity)
    reused = dataclass_replace(identity,
                               start_time_ticks=identity.start_time_ticks + 1)
    assert not _OwnedSubprocessSession._same_stable_identity(identity, reused)


def test_subprocess_capability_preflight_rejects_blocked_pidfd_signal(
        monkeypatch):
    close_spy = mock.Mock(wraps=os.close)
    monkeypatch.setattr('pool.activities.os.close', close_spy)
    monkeypatch.setattr(
        'pool.activities.pidfd_open',
        lambda _pid: os.open('/dev/null', os.O_RDONLY),
    )
    monkeypatch.setattr(
        'pool.activities.pidfd_send_signal',
        mock.Mock(side_effect=PermissionError('blocked by seccomp')),
    )

    assert not _subprocess_supervisor_capabilities_available()
    close_spy.assert_called_once()


@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='pidfds require Linux')
def test_subprocess_preflight_uses_wrappers_when_native_missing(monkeypatch):
    monkeypatch.delattr(activities.os, 'pidfd_open', raising=False)
    monkeypatch.delattr(activities.signal, 'pidfd_send_signal', raising=False)

    assert _subprocess_supervisor_capabilities_available()


@pytest.mark.asyncio
async def test_handshake_ack_fd_is_closed_exactly_once(monkeypatch):
    """Handshake abort cannot close a later user of a recycled fd number."""
    real_pipe = os.pipe
    real_close = os.close
    pipes = []
    ack_close_calls = []

    def tracked_pipe():
        result = real_pipe()
        pipes.append(result)
        return result

    def tracked_close(fd):
        if len(pipes) >= 2 and fd == pipes[1][1]:
            ack_close_calls.append(fd)
        return real_close(fd)

    monkeypatch.setattr('pool.activities.os.pipe', tracked_pipe)
    monkeypatch.setattr('pool.activities.os.close', tracked_close)
    monkeypatch.setattr(
        'pool.activities._subprocess_supervisor_capabilities_available',
        lambda: True)
    monkeypatch.setattr(
        'pool.activities._OwnedSubprocessSession.capture',
        mock.Mock(side_effect=RuntimeError('synthetic capture failure')),
    )

    with pytest.raises(RuntimeError, match='synthetic capture failure'):
        await _launch_owned_subprocess(
            [sys.executable, '-c', 'raise SystemExit(0)'],
            dict(os.environ),
            'kumo-pred',
            'job-1',
        )

    # asyncio may allocate additional subprocess pipes after the two ownership
    # handshake pipes; the second pipe's write end is the acknowledgement fd.
    assert len(pipes) >= 2
    assert ack_close_calls == [pipes[1][1]]


@pytest.mark.asyncio
async def test_cancellation_remains_primary_when_session_cleanup_fails(
        monkeypatch):
    fake_proc = _FakeProc(exit_after_term=False, exit_after_kill=True)
    owned_session = SimpleNamespace(
        terminate=mock.AsyncMock(side_effect=RuntimeError('terminate failed')),
        wait_for_quiescence=mock.AsyncMock(
            side_effect=RuntimeError('quiescence failed')),
    )
    monkeypatch.setattr(
        'pool.activities._launch_owned_subprocess',
        mock.AsyncMock(return_value=(fake_proc, owned_session)),
    )

    task = asyncio.create_task(
        _run_kumo_subprocess('kumo-pred', 'job-1', '/tmp/cfg.json', None))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        _ = await task
    assert fake_proc.kill_called


# ----------------------------------------------------------------------
# GPU semaphore: concurrent train/predict serialized by per-instance
# semaphore so two activities don't fight over the GPU.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpu_semaphore_serializes_concurrent_train():
    """``train`` holds the GPU semaphore for the duration of the
    subprocess. Two concurrent ``train`` calls on the same instance
    must therefore run sequentially, not in parallel.
    """
    instance = _make_instance()

    # Track entry/exit order to assert serialization.
    events: list[str] = []

    async def _fake_subprocess(_binary, _job_id, _config_path, _store):
        events.append('start')
        # Yield long enough for the second task to definitely have
        # been scheduled and parked on the semaphore.
        await asyncio.sleep(0.05)
        events.append('end')
        return 'SUCCEEDED'

    async def _fake_ensure_snapshot(*_args, **_kwargs):
        return None  # residency check no-op for this unit test

    with mock.patch('pool.activities._run_kumo_subprocess',
                    _fake_subprocess), \
            mock.patch.object(instance, '_ensure_snapshot',
                              _fake_ensure_snapshot), \
            mock.patch(
                'pool.activities.restore_job_sidecars_for_use_from_config_path',
                return_value=SimpleNamespace(cleanup=lambda: None)):
        results = await asyncio.gather(
            instance.train('graph-1bf8d6a669f6a6caba68a2393f1044e3@v1', 'job-1',
                           '/tmp/cfg-a.json'),
            instance.train('graph-44a6dcdb089c6e81b7171d8a74a7da57@v1', 'job-2',
                           '/tmp/cfg-b.json'),
        )

    assert results == ['SUCCEEDED', 'SUCCEEDED']
    # Strict serialization: every 'start' must be followed by an 'end'
    # before the next 'start'. Interleaving (start, start, end, end)
    # would mean the semaphore failed to serialize.
    assert events == ['start', 'end', 'start',
                      'end'], (f'GPU semaphore failed to serialize: {events}')


@pytest.mark.asyncio
async def test_gpu_semaphore_serializes_train_and_predict(tmp_path):
    """Train and predict share the same GPU semaphore — a predict job
    queued behind a running train job must wait for train to finish."""
    instance = _make_instance()

    events: list[str] = []

    async def _fake_subprocess(binary, _job_id, _config_path, _store):
        events.append(f'{binary}:start')
        await asyncio.sleep(0.05)
        events.append(f'{binary}:end')
        return 'FAILED' if binary == 'kumo-pred' else 'SUCCEEDED'

    async def _fake_ensure_snapshot(*_args, **_kwargs):
        return None

    prediction_config_path = tmp_path / 'pred.json'
    prediction_config_path.write_text('{}')
    prediction_config = _make_prediction_config(str(tmp_path / 'output'))

    with mock.patch('pool.activities._run_kumo_subprocess',
                    _fake_subprocess), \
            mock.patch.object(instance, '_ensure_snapshot',
                              _fake_ensure_snapshot), \
            mock.patch('pool.activities._shard_pred_config',
                       return_value=str(prediction_config_path)), \
            mock.patch('kumo_core.job_config.PredictionConfig.load',
                       return_value=prediction_config), \
            mock.patch(
                'pool.activities.restore_job_sidecars_for_use_from_config_path',
                return_value=SimpleNamespace(cleanup=lambda: None)):
        await asyncio.gather(
            instance.train('graph-1bf8d6a669f6a6caba68a2393f1044e3@v1', 'job-1',
                           '/tmp/cfg-a.json'),
            instance.predict('graph-44a6dcdb089c6e81b7171d8a74a7da57@v1',
                             'job-2', str(prediction_config_path),
                             job_input=_make_prediction_job_input()),
        )

    # Whichever ran first, the other one must not interleave.
    train_end_idx = events.index('kumo-auto:end')
    pred_start_idx = events.index('kumo-pred:start')
    train_start_idx = events.index('kumo-auto:start')
    pred_end_idx = events.index('kumo-pred:end')
    if train_start_idx < pred_start_idx:
        assert train_end_idx < pred_start_idx, (
            f'train should end before predict starts: {events}')
    else:
        assert pred_end_idx < train_start_idx, (
            f'predict should end before train starts: {events}')


@pytest.mark.asyncio
async def test_gpu_slots_pin_concurrent_processes(tmp_path, monkeypatch):
    # Each run records its GPU, then waits for the other to start.
    probe = tmp_path / 'bin' / 'kumo-gpu-probe'
    probe.parent.mkdir()
    probe.write_text('#!/bin/sh\n'
                     'echo "$CUDA_VISIBLE_DEVICES" > "$2"\n'
                     'for _ in $(seq 100); do\n'
                     '  [ "$(ls "$(dirname "$2")" | wc -l)" -ge 2 ] && exit 0\n'
                     '  sleep 0.1\n'
                     'done\n'
                     'exit 1\n')
    probe.chmod(0o755)
    monkeypatch.setenv('PATH', f'{probe.parent}:{os.environ["PATH"]}')
    seen = tmp_path / 'seen'
    seen.mkdir()
    slots = _GpuSlots(2, devices=['2', '3'])

    async def run(i: int) -> str:
        async with slots.acquire() as environment:
            return await _run_kumo_subprocess(
                'kumo-gpu-probe', f'job-{i}', str(seen / str(i)), None,
                subprocess_environment=environment)

    assert await asyncio.gather(run(0), run(1)) == ['SUCCEEDED'] * 2
    assert sorted(f.read_text().strip() for f in seen.iterdir()) == ['2', '3']


def test_gpu_slots_need_one_gpu_per_slot():
    with pytest.raises(ValueError, match='sees 1 GPU'):
        _GpuSlots(2, devices=['0'])


@pytest.mark.skipif(
    subprocess.run('nvidia-smi -L', shell=True, capture_output=True,
                   check=False).returncode != 0, reason='needs a GPU')
def test_visible_gpus_stop_at_invalid_entry():
    result = subprocess.run([
        sys.executable, '-c',
        'from pool.activities import _visible_gpus; print(_visible_gpus())'
    ], env={
        **os.environ, 'CUDA_VISIBLE_DEVICES': '0,-1'
    }, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                            check=True)
    assert result.stdout.strip() == "['0']"


@pytest.mark.asyncio
async def test_train_restores_and_cleans_ttng_sidecars(monkeypatch):
    instance = _make_instance()

    async def _fake_ensure_snapshot(*_args, **_kwargs):
        return None

    async def _fake_subprocess(_binary, _job_id, _config_path, _store):
        return 'SUCCEEDED'

    async def _run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    sidecars = mock.MagicMock()
    with mock.patch.object(instance, '_ensure_snapshot',
                           _fake_ensure_snapshot), \
            mock.patch('pool.activities._run_kumo_subprocess',
                       _fake_subprocess), \
            mock.patch('pool.activities.asyncio.to_thread', _run_inline), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value='/persist'), \
            mock.patch(
                'pool.activities.restore_job_sidecars_for_use_from_config_path',
                return_value=sidecars) as restore:
        await instance.train(_TEST_SNAPSHOT_ID, 'train-job', '/tmp/train.json')

    restore.assert_called_once_with(
        _TEST_SNAPSHOT_ID,
        '/tmp/train.json',
        prediction=False,
        data_persist_dir='/persist',
    )
    sidecars.cleanup.assert_called_once_with()


@pytest.mark.asyncio
async def test_train_sidecar_restore_failure_hard_fails(monkeypatch):
    """Option A: the sidecar hand-off is authoritative. If materialize uploaded
    the per-job train_tables extension but restore cannot land it on this pod,
    that is a real bug -- the train activity must fail loud (propagate), NOT
    silently proceed. Temporal retries transient restore failures."""
    instance = _make_instance()

    async def _fake_ensure_snapshot(*_a, **_k):
        return None

    async def _fake_subprocess(_binary, _job_id, _config_path, _store):
        return 'SUCCEEDED'

    def _raise_restore(*_a, **_k):
        raise FileNotFoundError('sidecar not landed on this pod')

    async def _run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    subprocess_mock = mock.MagicMock(side_effect=_fake_subprocess)

    with mock.patch.object(instance, '_ensure_snapshot',
                           _fake_ensure_snapshot), \
            mock.patch('pool.activities._run_kumo_subprocess',
                       subprocess_mock), \
            mock.patch('pool.activities.asyncio.to_thread', _run_inline), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value='/persist'), \
            mock.patch(
                'pool.activities.restore_job_sidecars_for_use_from_config_path',
                side_effect=_raise_restore):
        with pytest.raises(FileNotFoundError):
            await instance.train(_TEST_SNAPSHOT_ID, 'train-job',
                                 '/tmp/train.json')

    # Fail loud: the subprocess must never run when the sidecar is missing.
    subprocess_mock.assert_not_called()


@pytest.mark.asyncio
async def test_predict_sidecar_restore_failure_hard_fails(
        monkeypatch, tmp_path):
    """Option A (predict): a failed predict-stage sidecar restore must propagate
    -- no silent proceed into a subprocess that would crash downstream."""
    instance = _make_instance()

    async def _fake_ensure_snapshot(*_a, **_k):
        return None

    async def _fake_subprocess(_binary, _job_id, _config_path, _store):
        return 'SUCCEEDED'

    def _raise_restore(*_a, **_k):
        raise FileNotFoundError('predict sidecar not landed on this pod')

    async def _run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    subprocess_mock = mock.MagicMock(side_effect=_fake_subprocess)
    prediction_config_path = tmp_path / 'pred.json'
    prediction_config_path.write_text('{}')
    prediction_config = _make_prediction_config(str(tmp_path / 'output'))

    with mock.patch.object(instance, '_ensure_snapshot',
                           _fake_ensure_snapshot), \
            mock.patch('pool.activities._run_kumo_subprocess',
                       subprocess_mock), \
            mock.patch('pool.activities._shard_pred_config',
                       return_value=str(prediction_config_path)), \
            mock.patch('kumo_core.job_config.PredictionConfig.load',
                       return_value=prediction_config), \
            mock.patch('pool.activities.asyncio.to_thread', _run_inline), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value='/persist'), \
            mock.patch(
                'pool.activities.restore_job_sidecars_for_use_from_config_path',
                side_effect=_raise_restore):
        with pytest.raises(FileNotFoundError):
            await instance.predict(_TEST_SNAPSHOT_ID, 'bp-job',
                                   str(prediction_config_path),
                                   job_input=_make_prediction_job_input())

    subprocess_mock.assert_not_called()


@pytest.mark.asyncio
async def test_predict_merges_bounded_stats_into_ownership_manifest(
        monkeypatch, tmp_path):
    import json

    from kumo_core.train.artifact_stats import (
        PREDICTION_ARTIFACT_STATS_PATH_ENV,
        PREDICTION_ARTIFACT_STATS_REQUEST_ENV,
        PredictionArtifactStatsPayload,
        PredictionArtifactStatsRequest,
        ShardArtifactStats,
    )
    from kumo_core.types import PredictionArtifactType

    from kumo.train.artifact_manifest import load_shard_manifest

    instance = _make_instance()
    instance._ensure_snapshot = mock.AsyncMock()  # type: ignore[method-assign]
    source_config = _make_prediction_config(str(tmp_path / 'output'))
    generation = (tmp_path / 'output' / '_attempts' / 'runtime-00000' /
                  'producer-attempt')
    runtime_config = _make_prediction_config(str(generation))
    config_path = tmp_path / 'prediction.json'
    config_path.write_text('{}')
    captured_environment = None

    async def fake_subprocess(_binary, _job_id, _config_path, _store, **kwargs):
        nonlocal captured_environment
        captured_environment = kwargs['subprocess_environment']
        request = PredictionArtifactStatsRequest.from_dict(
            json.loads(
                captured_environment[PREDICTION_ARTIFACT_STATS_REQUEST_ENV]))
        output = (Path(request.output_dir) /
                  PredictionArtifactType.PREDICTIONS.output_file_name)
        output.mkdir(parents=True)
        part = output / 'part-00000-00000.parquet'
        part.write_bytes(b'owned-output')
        artifact = ShardArtifactStats(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            output_prefix=str(output),
            partition_count=1,
            schema_fingerprint='a' * 64,
            num_rows=1,
            num_cols=3,
            sample_rows=1,
            sample_prefix_logical_bytes=(0, 10),
            size_bytes=10,
            physical_size_bytes=part.stat().st_size,
            file_count=1,
        )
        payload = PredictionArtifactStatsPayload(request, 1, (artifact, ))
        payload.validate(request)
        Path(captured_environment[PREDICTION_ARTIFACT_STATS_PATH_ENV]
             ).write_text(payload.to_json())
        return JobState.SUCCEEDED

    monkeypatch.setattr(activities.uuid, 'uuid4',
                        lambda: SimpleNamespace(hex='producer-attempt'))
    monkeypatch.setattr(
        activities, 'deploy_env', lambda: SimpleNamespace(
            data_compute=SimpleNamespace(data_root=str(tmp_path / 'data'))))
    monkeypatch.setattr(
        'kumo_core.job_config.PredictionConfig.load',
        mock.MagicMock(side_effect=[source_config, runtime_config]))
    monkeypatch.setattr(activities, '_shard_pred_config',
                        lambda *_args, **_kwargs: str(config_path))
    monkeypatch.setattr(
        'kumo.train.artifact_manifest.hash_prediction_config',
        lambda _path: 'b' * 64,
    )
    monkeypatch.setattr(activities, '_run_kumo_subprocess', fake_subprocess)
    sidecars = SimpleNamespace(cleanup=lambda: None)
    monkeypatch.setattr(
        activities,
        'restore_job_sidecars_for_use_from_config_path',
        lambda *_args, **_kwargs: sidecars,
    )

    result = await instance.predict(
        _TEST_SNAPSHOT_ID,
        'bp-job',
        str(config_path),
        job_input=_make_prediction_job_input(),
        artifact_ownership_enabled=True,
        producer_stats_enabled=True,
    )

    assert result.state is JobState.SUCCEEDED
    assert result.artifact_manifest is not None
    manifest = load_shard_manifest(result.artifact_manifest)
    assert manifest.producer_stats is True
    assert manifest.artifacts[0].num_rows == 1
    assert manifest.artifacts[0].file_count == 1
    assert captured_environment is not None
    assert not list(tmp_path.glob('data/kumo-pred-stats-*'))


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_during_subprocess', [False, True])
async def test_predict_stats_cleanup_survives_repeated_cancellation(
        monkeypatch, tmp_path, cancel_during_subprocess):
    instance = _make_instance()
    instance._ensure_snapshot = mock.AsyncMock()  # type: ignore[method-assign]
    source_config = _make_prediction_config(str(tmp_path / 'output'))
    generation = (tmp_path / 'output' / '_attempts' / 'runtime-00000' /
                  'producer-attempt')
    runtime_config = _make_prediction_config(str(generation))
    config_path = tmp_path / 'prediction.json'
    config_path.write_text('{}')
    subprocess_started = asyncio.Event()
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    cleanup_finished = threading.Event()
    real_rmtree = shutil.rmtree

    async def fake_subprocess(*_args, **_kwargs):
        subprocess_started.set()
        if cancel_during_subprocess:
            await asyncio.Event().wait()
        return JobState.FAILED

    def blocking_rmtree(path, *args, **kwargs):
        if Path(path).name.startswith('kumo-pred-stats-'):
            cleanup_started.set()
            assert release_cleanup.wait(timeout=10)
            try:
                return real_rmtree(path, *args, **kwargs)
            finally:
                cleanup_finished.set()
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(activities.uuid, 'uuid4',
                        lambda: SimpleNamespace(hex='producer-attempt'))
    monkeypatch.setattr(
        activities, 'deploy_env', lambda: SimpleNamespace(
            data_compute=SimpleNamespace(data_root=str(tmp_path / 'data'))))
    monkeypatch.setattr(
        'kumo_core.job_config.PredictionConfig.load',
        mock.MagicMock(side_effect=[source_config, runtime_config]))
    monkeypatch.setattr(activities, '_shard_pred_config',
                        lambda *_args, **_kwargs: str(config_path))
    monkeypatch.setattr(
        'kumo.train.artifact_manifest.hash_prediction_config',
        lambda _path: 'b' * 64,
    )
    monkeypatch.setattr(activities, '_run_kumo_subprocess', fake_subprocess)
    monkeypatch.setattr(activities.shutil, 'rmtree', blocking_rmtree)
    monkeypatch.setattr(
        activities,
        'restore_job_sidecars_for_use_from_config_path',
        lambda *_args, **_kwargs: SimpleNamespace(cleanup=lambda: None),
    )

    task = asyncio.create_task(
        instance.predict(
            _TEST_SNAPSHOT_ID,
            'bp-job',
            str(config_path),
            job_input=_make_prediction_job_input(),
            artifact_ownership_enabled=True,
            producer_stats_enabled=True,
        ))
    await subprocess_started.wait()
    if cancel_during_subprocess:
        task.cancel()
    await _wait_for_thread_event(cleanup_started)
    task.cancel()
    await asyncio.sleep(0)
    cleanup_was_joined = not task.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup_was_joined
    assert cleanup_finished.is_set()
    assert not list(tmp_path.glob('data/kumo-pred-stats-*'))


# ----------------------------------------------------------------------
# Snapshot mutation invariant: _ensure_snapshot dispatcher
# ----------------------------------------------------------------------

_TEST_SNAPSHOT_ID = 'graph-1bf8d6a669f6a6caba68a2393f1044e3@v1'


@pytest.mark.asyncio
async def test_ensure_snapshot_dispatcher_inline_when_no_temporal_client():
    """temporal_client=None ⇒ run inline."""
    instance = _make_instance()
    captured: list = []

    async def _fake_local(snapshot_id, *, ingest_args=None):
        captured.append((snapshot_id, ingest_args))
        return mock.sentinel.IngestResult

    with mock.patch.object(instance, '_ensure_snapshot_local', _fake_local):
        result = await instance._ensure_snapshot(_TEST_SNAPSHOT_ID)

    assert captured == [(_TEST_SNAPSHOT_ID, None)]
    assert result is mock.sentinel.IngestResult


@pytest.mark.asyncio
async def test_ensure_snapshot_dispatcher_delegates_when_temporal_client_set():
    """temporal_client set ⇒ delegate to daemon, inline body must not run."""
    fake_inner_client = mock.MagicMock()
    fake_inner_client.execute_workflow = mock.AsyncMock(return_value=None)
    fake_temporal_client = mock.MagicMock()
    fake_temporal_client.get = mock.AsyncMock(return_value=fake_inner_client)

    instance = SnapshotActivities(
        pool_worker_uid='pool-worker-nodeA',
        controller=_MockController(),
        storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
        temporal_client=fake_temporal_client,
    )

    inline_called = False

    async def _should_not_run(*_args, **_kwargs):
        nonlocal inline_called
        inline_called = True

    fake_info = mock.MagicMock(workflow_id='wf-1', activity_id='1')
    with mock.patch.object(instance, '_ensure_snapshot_local',
                           _should_not_run), \
            mock.patch('pool.activities.activity.info',
                       return_value=fake_info):
        result = await instance._ensure_snapshot(_TEST_SNAPSHOT_ID)

    assert result is None
    assert not inline_called

    from temporalio_commons.worker_registry import RunSingleActivityWorkflow
    fake_inner_client.execute_workflow.assert_awaited_once()
    call = fake_inner_client.execute_workflow.await_args
    assert call.args[0] is RunSingleActivityWorkflow.run
    assert call.kwargs['args'] == ['ensure_snapshot', [_TEST_SNAPSHOT_ID]]
    assert call.kwargs['task_queue'] == 'diskgraph-daemon-pool-worker-nodeA'


@pytest.mark.asyncio
async def test_publish_training_progress_reads_off_event_loop(monkeypatch):
    """``publish_training_progress`` must run its blocking progress read off
    the event loop via ``asyncio.to_thread``; otherwise a slow S3 read pins
    the loop.
    """
    def _fake_read(_job_config_path):
        return None  # no progress yet -> publish returns False

    monkeypatch.setattr('pool.training_progress._read_training_progress_output',
                        _fake_read)
    to_thread_call = {}

    async def _fake_to_thread(func, *args, **kwargs):
        to_thread_call['func'] = func
        to_thread_call['args'] = args
        to_thread_call['kwargs'] = kwargs
        return func(*args, **kwargs)

    monkeypatch.setattr('pool.training_progress.asyncio.to_thread',
                        _fake_to_thread)

    job_store = mock.Mock(spec=MLJobOutputStore)
    result = await publish_training_progress('job-1', '/tmp/cfg.json',
                                             job_store)

    assert result is False
    assert to_thread_call == {
        'func': _fake_read,
        'args': ('/tmp/cfg.json', ),
        'kwargs': {},
    }


@pytest.mark.asyncio
async def test_ensure_snapshot_dispatcher_rejects_ingest_args_when_delegating():
    """Daemon-mode ensure (ingest_args set) must not reach the delegate
    branch."""
    from pool.diskgraph_args import DiskgraphIngestArgs

    instance = SnapshotActivities(
        pool_worker_uid='pool-worker-nodeA',
        controller=_MockController(),
        storage_mode=PoolStorageMode.ACTIVITY_ON_PVC,
        temporal_client=mock.MagicMock(),
    )

    with pytest.raises(AssertionError, match='must never run daemon-mode'):
        await instance._ensure_snapshot(
            _TEST_SNAPSHOT_ID,
            ingest_args=mock.MagicMock(spec=DiskgraphIngestArgs),
        )


# ----------------------------------------------------------------------
# _ensure_snapshot_local: SSD miss → dataplane restore → fresh ingest.
#
# Branches covered here:
#   0. SSD hit → daemon reports SNAPSHOT_READY with size for backfill
#   1. SSD clean-miss → dataplane restore success → return restored result
#   2. SSD clean-miss + dataplane miss + daemon mode → fresh ingest
#   3. SSD clean-miss + dataplane miss + consumer mode (no ingest_args)
#      → ApplicationError(non_retryable=True, type=PERMANENT_SNAPSHOT_MISS)
# ----------------------------------------------------------------------


def test_diskgraph_ingest_config_strips_hydrated_connectors_before_subprocess(
        monkeypatch):
    from types import SimpleNamespace

    from pool.activities import _build_diskgraph_ingest_config

    graph = mock.MagicMock()
    graph_config = SimpleNamespace(
        data_source_dict={'connector_id': mock.sentinel.dynamic_config}, )
    monkeypatch.setattr(
        'rest.driver.graph_logic._build_graph_from_definition_with_specs',
        lambda *_args, **_kwargs: graph,
    )
    monkeypatch.setattr(
        'rest.driver.data_snapshot_api_driver._build_ingest_graph_config',
        lambda *_args, **_kwargs: graph_config,
    )
    ingest_args = SimpleNamespace(
        all_graph_tables={},
        graph_definition=SimpleNamespace(col_groups=[]),
        graph_display_name='graph',
        connector_specs={},
        table_data_versions={},
    )

    result = _build_diskgraph_ingest_config(ingest_args)

    assert result is graph_config
    assert result.data_source_dict == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'storage_mode,expected_event_count',
    [(PoolStorageMode.DAEMON_ON_PVC, 1),
     (PoolStorageMode.ACTIVITY_ON_EMPTYDIR, 0)],
)
async def test_ensure_snapshot_local_ssd_hit_reports_size_for_daemon_only(
        tmp_path, storage_mode, expected_event_count):
    """Daemon SSD hits backfill size; activity-scoped pods stay transient."""
    from kumo.diskgraph import IngestResult

    instance = _make_instance(storage_mode=storage_mode)
    instance._send_status_report = mock.AsyncMock()

    local_dir = tmp_path / 'cached-local'
    local_dir.mkdir()
    (local_dir / 'payload.bin').write_bytes(b'abc12345')
    cached = IngestResult(
        index_path=str(tmp_path / 'idx'),
        col_stats_path=str(tmp_path / 'col'),
        ingest_stats_path=str(tmp_path / 'stats'),
        kumo_stats_dir=str(tmp_path / 'kumo'),
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.has_resumable_ingest.return_value = False
    fake_service.verify_integrity.return_value = cached

    with mock.patch('pool.activities.get_service',
                    return_value=fake_service), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')):
        result = await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)

    assert result is cached
    fake_service.verify_integrity.assert_called_once()
    assert instance._send_status_report.await_count == expected_event_count
    if expected_event_count:
        event = instance._send_status_report.await_args.kwargs['event']
        assert event.type == WorkerEventType.SNAPSHOT_READY
        assert event.size_bytes == 8


@pytest.mark.asyncio
async def test_ensure_snapshot_local_invalidates_and_deletes_corrupt_ssd(
        tmp_path):
    """A verified corrupt L1 is invalidated upstream and removed locally."""
    from temporalio.exceptions import ApplicationError

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    instance._ready_snapshots.add(_TEST_SNAPSHOT_ID)
    local_dir = tmp_path / 'corrupt-local'
    local_dir.mkdir()
    (local_dir / 'partial.bin').write_bytes(b'partial')
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'missing-dataplane')
    fake_service.has_resumable_ingest.return_value = False

    def raise_corruption(_):
        raise _DiskgraphCorruptionError('bad local manifest')

    async def run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    fake_service.verify_integrity = raise_corruption
    fake_service.restore_snapshot.return_value = None

    with (
            mock.patch.dict(sys.modules, {'diskgraph': _FAKE_DISKGRAPH}),
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
    ):
        with pytest.raises(ApplicationError):
            await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)

    assert not local_dir.exists()
    assert _TEST_SNAPSHOT_ID not in instance._ready_snapshots
    events = [
        call.kwargs['event']
        for call in instance._send_status_report.await_args_list
    ]
    assert [event.type for event in events] == [
        WorkerEventType.SSD_CACHE_CORRUPTED,
        WorkerEventType.SNAPSHOT_RESTORE_STARTED,
    ]
    assert events[0].snapshot_id == _TEST_SNAPSHOT_ID


@pytest.mark.asyncio
async def test_corrupt_ssd_invalidation_survives_cleanup_cancellation(tmp_path):
    """Cancellation cannot leave a deleted L1 advertised as READY."""
    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    instance._ready_snapshots.add(_TEST_SNAPSHOT_ID)
    local_dir = tmp_path / 'corrupt-local'
    local_dir.mkdir()
    (local_dir / 'partial.bin').write_bytes(b'partial')

    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.has_resumable_ingest.return_value = False
    fake_service.verify_integrity.side_effect = _DiskgraphCorruptionError(
        'bad local manifest')

    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    allow_cleanup = threading.Event()
    loop = asyncio.get_running_loop()
    real_rmtree = shutil.rmtree

    def blocking_rmtree(*args, **kwargs):
        loop.call_soon_threadsafe(cleanup_started.set)
        assert allow_cleanup.wait(timeout=5)
        try:
            return real_rmtree(*args, **kwargs)
        finally:
            loop.call_soon_threadsafe(cleanup_finished.set)

    with (
            mock.patch.dict(sys.modules, {'diskgraph': _FAKE_DISKGRAPH}),
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities.shutil.rmtree',
                       side_effect=blocking_rmtree),
    ):
        ensure_task = asyncio.create_task(
            instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID))
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        ensure_task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await ensure_task
        finally:
            allow_cleanup.set()
        await asyncio.wait_for(cleanup_finished.wait(), timeout=5)

    assert not local_dir.exists()
    assert _TEST_SNAPSHOT_ID not in instance._ready_snapshots
    events = [
        call.kwargs['event']
        for call in instance._send_status_report.await_args_list
    ]
    assert [event.type for event in events] == [
        WorkerEventType.SSD_CACHE_CORRUPTED,
    ]
    fake_service.restore_snapshot.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup_stage', ['local', 'sidecar'])
async def test_corrupt_ssd_cleanup_failure_keeps_corruption_primary(
        tmp_path, cleanup_stage):
    """Cleanup failure is secondary after authoritative L1 invalidation."""
    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    instance._ready_snapshots.add(_TEST_SNAPSHOT_ID)
    local_dir = tmp_path / 'corrupt-local'
    local_dir.mkdir()
    (local_dir / 'partial.bin').write_bytes(b'partial')

    corruption_error = _DiskgraphCorruptionError('bad local manifest')
    cleanup_error = OSError(f'{cleanup_stage} cleanup failed')
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.has_resumable_ingest.return_value = False
    fake_service.verify_integrity.side_effect = corruption_error
    if cleanup_stage == 'sidecar':
        fake_service.cleanup_ingest_sidecars.side_effect = cleanup_error

    real_rmtree = shutil.rmtree

    def fail_selected_cleanup(*args, **kwargs):
        if cleanup_stage == 'local':
            raise cleanup_error
        return real_rmtree(*args, **kwargs)

    with (
            mock.patch.dict(sys.modules, {'diskgraph': _FAKE_DISKGRAPH}),
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities.shutil.rmtree',
                       side_effect=fail_selected_cleanup),
    ):
        with pytest.raises(_DiskgraphCorruptionError) as exc_info:
            await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)

    assert exc_info.value is corruption_error
    assert exc_info.value.__cause__ is cleanup_error
    assert _TEST_SNAPSHOT_ID not in instance._ready_snapshots
    events = [
        call.kwargs['event']
        for call in instance._send_status_report.await_args_list
    ]
    assert [event.type for event in events] == [
        WorkerEventType.SSD_CACHE_CORRUPTED,
    ]
    fake_service.restore_snapshot.assert_not_called()
    fake_service.ingest.assert_not_called()
    if cleanup_stage == 'local':
        assert local_dir.exists()
        fake_service.cleanup_ingest_sidecars.assert_not_called()
    else:
        assert not local_dir.exists()
        fake_service.cleanup_ingest_sidecars.assert_called_once()


@pytest.mark.asyncio
async def test_corrupt_ssd_cleanup_tolerates_concurrent_local_removal(tmp_path):
    """A concurrent local delete is idempotent and recovery keeps moving."""
    from temporalio.exceptions import ApplicationError

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    instance._ready_snapshots.add(_TEST_SNAPSHOT_ID)
    local_dir = tmp_path / 'corrupt-local'
    local_dir.mkdir()
    (local_dir / 'partial.bin').write_bytes(b'partial')

    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'missing-dataplane')
    fake_service.has_resumable_ingest.return_value = False
    fake_service.verify_integrity.side_effect = _DiskgraphCorruptionError(
        'bad local manifest')
    fake_service.restore_snapshot.return_value = None

    real_rmtree = shutil.rmtree

    def remove_then_report_absent(*args, **kwargs):
        real_rmtree(*args, **kwargs)
        raise FileNotFoundError(str(local_dir))

    with (
            mock.patch.dict(sys.modules, {'diskgraph': _FAKE_DISKGRAPH}),
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities.shutil.rmtree',
                       side_effect=remove_then_report_absent),
    ):
        with pytest.raises(ApplicationError):
            await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)

    assert _TEST_SNAPSHOT_ID not in instance._ready_snapshots
    assert not local_dir.exists()
    fake_service.cleanup_ingest_sidecars.assert_called_once()
    fake_service.restore_snapshot.assert_called_once()
    fake_service.ingest.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('storage_mode', 'ingest_args_present', 'restore_error'),
    [
        (PoolStorageMode.DAEMON_ON_PVC, True,
         _DiskgraphCorruptionError('copied-local manifest mismatch')),
        (PoolStorageMode.DAEMON_ON_PVC, False,
         ConnectionError('copy interrupted')),
        (PoolStorageMode.ACTIVITY_ON_PVC, False,
         _DiskgraphCorruptionError('copied-local manifest mismatch')),
        (PoolStorageMode.ACTIVITY_ON_EMPTYDIR, False,
         _DiskgraphCorruptionError('copied-local manifest mismatch')),
        (PoolStorageMode.ACTIVITY_ON_PVC, False,
         OSError('local disk write failed')),
    ],
)
async def test_ensure_snapshot_local_restore_failure_is_non_destructive(
        tmp_path, storage_mode, ingest_args_present, restore_error):
    """An ambiguous copied-local failure is retryable and cannot mutate L2."""
    instance = _make_instance(storage_mode=storage_mode)
    instance._send_status_report = mock.AsyncMock()
    instance._ready_snapshots.add(_TEST_SNAPSHOT_ID)
    local_dir = tmp_path / 'partial-restore'
    dataplane_dir = tmp_path / 'trusted-dataplane'
    dataplane_dir.mkdir()
    dataplane_payload = dataplane_dir / 'snapshot.bin'
    dataplane_payload.write_bytes(b'trusted-L2')

    def _corrupt_restore(*_args, **_kwargs):
        local_dir.mkdir()
        (local_dir / 'partial.bin').write_bytes(b'partial')
        raise restore_error

    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(dataplane_dir)
    fake_service.restore_snapshot.side_effect = _corrupt_restore
    ingest_args = mock.MagicMock() if ingest_args_present else None

    with (
            mock.patch.dict(sys.modules, {'diskgraph': _FAKE_DISKGRAPH}),
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities.fsutil.rm') as remote_rm,
    ):
        with pytest.raises(RuntimeError,
                           match='failed to restore snapshot') as excinfo:
            await instance._ensure_snapshot_local(
                _TEST_SNAPSHOT_ID,
                ingest_args=ingest_args,
            )

        error_text = str(excinfo.value)
        assert all(part in error_text
                   for part in (_TEST_SNAPSHOT_ID, str(dataplane_dir),
                                str(local_dir), str(restore_error)))
        assert excinfo.value.__cause__ is restore_error
        assert not is_error_non_retryable(excinfo.value)
        assert not local_dir.exists()
        assert _TEST_SNAPSHOT_ID not in instance._ready_snapshots
        assert dataplane_payload.read_bytes() == b'trusted-L2'

    remote_rm.assert_not_called()
    fake_service.restore_snapshot.assert_called_once()
    fake_service.ingest.assert_not_called()
    events = [
        call.kwargs['event']
        for call in instance._send_status_report.await_args_list
    ]
    assert [event.type for event in events] == [
        WorkerEventType.SNAPSHOT_RESTORE_STARTED,
    ]


@pytest.mark.asyncio
async def test_ensure_snapshot_local_restores_from_dataplane_on_ssd_miss(
        tmp_path):
    """SSD has no local dir → service.restore_snapshot returns a hit →
    return that result without touching the fresh-ingest path."""
    from kumo.diskgraph import IngestResult
    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()

    fake_restored = IngestResult(
        index_path=str(tmp_path / 'idx'),
        col_stats_path=str(tmp_path / 'col'),
        ingest_stats_path=str(tmp_path / 'stats'),
        kumo_stats_dir=str(tmp_path / 'kumo'),
    )
    local_dir = tmp_path / 'restored-local'

    def _restore_snapshot(*_args, **_kwargs):
        local_dir.mkdir()
        (local_dir / 'payload.bin').write_bytes(b'abc123')
        return fake_restored

    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.restore_snapshot.side_effect = _restore_snapshot

    with mock.patch('pool.activities.get_service',
                    return_value=fake_service), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')):
        result = await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)

    assert result is fake_restored
    fake_service.restore_snapshot.assert_called_once()
    fake_service.ingest.assert_not_called()
    events = [
        call.kwargs['event']
        for call in instance._send_status_report.await_args_list
    ]
    assert [event.type for event in events] == [
        WorkerEventType.SNAPSHOT_RESTORE_STARTED,
        WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
        WorkerEventType.SNAPSHOT_READY,
    ]
    assert events[1].size_bytes == 6
    assert events[2].size_bytes == 6


@pytest.mark.asyncio
async def test_ensure_snapshot_local_restore_passes_heartbeat(
        tmp_path, monkeypatch):
    """The restore branch has no wrapping ``@auto_heartbeater`` (unlike a
    normal activity), so ``service.restore_snapshot`` must get its own
    heartbeat callback wired through -- otherwise a cold worker restoring
    a large mirror blows the workflow's 5-minute heartbeat_timeout."""
    from kumo.diskgraph import IngestResult

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    schedule_mock = mock.Mock()
    monkeypatch.setattr('pool.activities._schedule_activity_heartbeat',
                        schedule_mock)

    fake_restored = IngestResult(
        index_path=str(tmp_path / 'idx'),
        col_stats_path=str(tmp_path / 'col'),
        ingest_stats_path=str(tmp_path / 'stats'),
        kumo_stats_dir=str(tmp_path / 'kumo'),
    )
    local_dir = tmp_path / 'restored-local'
    captured_kwargs = {}

    def _restore_snapshot(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        local_dir.mkdir()
        (local_dir / 'payload.bin').write_bytes(b'abc123')
        heartbeat = kwargs['heartbeat']
        heartbeat()
        heartbeat()
        return fake_restored

    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.restore_snapshot.side_effect = _restore_snapshot

    with mock.patch('pool.activities.get_service',
                    return_value=fake_service), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')):
        result = await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)

    assert result is fake_restored
    assert callable(captured_kwargs.get('heartbeat'))
    assert schedule_mock.call_count == 2


@pytest.mark.asyncio
async def test_ensure_snapshot_local_restore_joins_on_cancel(
        tmp_path, monkeypatch):
    """``to_thread`` cannot stop ``fsutil.copy`` once it is writing into
    ``local_dir``, so the restore dispatch -- like the neighbouring
    ``download_table_files`` and ``ingest`` dispatches -- must join the
    executor thread on cancellation instead of abandoning it."""
    from kumo.diskgraph import IngestResult

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    monkeypatch.setattr('pool.activities._schedule_activity_heartbeat',
                        mock.Mock())

    fake_restored = IngestResult(
        index_path=str(tmp_path / 'idx'),
        col_stats_path=str(tmp_path / 'col'),
        ingest_stats_path=str(tmp_path / 'stats'),
        kumo_stats_dir=str(tmp_path / 'kumo'),
    )
    local_dir = tmp_path / 'restored-local'

    def _restore_snapshot(*_args, **_kwargs):
        local_dir.mkdir()
        (local_dir / 'payload.bin').write_bytes(b'abc123')
        return fake_restored

    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.restore_snapshot.side_effect = _restore_snapshot

    restore_cancellation_args = []

    async def _capture_restore_kwargs(fn, *args, **kwargs):
        if fn is fake_service.restore_snapshot:
            restore_cancellation_args.append((
                kwargs.get('join_on_cancel'),
                kwargs.get('cancellation_cleanup_dir'),
            ))
        return await _run_diskgraph_async(fn, *args, **kwargs)

    with (
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities._run_diskgraph_async',
                       _capture_restore_kwargs),
    ):
        result = await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID)

    assert result is fake_restored
    assert restore_cancellation_args == [(True, str(local_dir))]


@pytest.mark.asyncio
async def test_ensure_snapshot_local_fresh_ingest_reports_snapshot_size(
        tmp_path):
    """Fresh daemon ingest records local snapshot size for overflow sizing."""
    from types import SimpleNamespace

    from kumo.diskgraph import IngestResult

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()

    local_dir = tmp_path / 'fresh-local'
    fake_result = IngestResult(
        index_path=str(tmp_path / 'idx'),
        col_stats_path=str(tmp_path / 'col'),
        ingest_stats_path=str(tmp_path / 'stats'),
        kumo_stats_dir=str(tmp_path / 'kumo'),
    )

    def _ingest(*_args, **_kwargs):
        local_dir.mkdir()
        (local_dir / 'snapshot.bin').write_bytes(b'payload')
        return fake_result

    table = mock.MagicMock()
    table.fingerprint.return_value = mock.sentinel.fingerprint
    ingest_args = SimpleNamespace(
        graph_config=mock.sentinel.graph_config,
        data_persist_dir=str(tmp_path / 'data-persist-dir'),
        table_data_versions={'transactions': 'v1'},
        connector_specs={},
        all_graph_tables={'transactions': table},
        source_size_bytes_by_table={},
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.restore_snapshot.return_value = None
    fake_service.prepare_direct_sources.return_value = {}
    fake_service.ingest.side_effect = _ingest

    async def _run_inline(fn, *args, **kwargs):
        kwargs.pop('cleanup_dir', None)
        kwargs.pop('abandonment_cleanup_dir', None)
        return fn(*args, **kwargs)

    with mock.patch('pool.activities.get_service',
                    return_value=fake_service), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')), \
            mock.patch('pool.activities.get_diskgraph_snapshot_ingest_path',
                       return_value=str(tmp_path / 'source')), \
            mock.patch('pool.activities._build_diskgraph_ingest_config',
                       return_value=mock.sentinel.graph_config), \
            mock.patch('pool.activities._run_diskgraph_async', _run_inline), \
            mock.patch('pool.activities.asyncio.to_thread', _run_inline):
        result = await instance._ensure_snapshot_local(
            _TEST_SNAPSHOT_ID,
            ingest_args=ingest_args,
        )

    assert result is fake_result
    fake_service.download_table_files.assert_called_once()
    fake_service.ingest.assert_called_once()
    events = [
        call.kwargs['event']
        for call in instance._send_status_report.await_args_list
    ]
    assert [event.type for event in events] == [
        WorkerEventType.SNAPSHOT_RESTORE_STARTED,
        WorkerEventType.SNAPSHOT_INGEST_STARTED,
        WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
        WorkerEventType.SNAPSHOT_READY,
    ]
    assert events[2].size_bytes == 7
    assert events[3].size_bytes == 7


@pytest.mark.asyncio
async def test_ensure_snapshot_local_mixes_direct_object_and_staged_sources(
        tmp_path):
    from types import SimpleNamespace

    from kumo.diskgraph import IngestResult

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    local_dir = tmp_path / 'fresh-object-source'
    source = {
        'type':
        'object_store_parquet',
        'objects': [{
            'path': 's3://bucket/direct/part.parquet',
            'size': 123,
            'e_tag': 'etag',
            'version': None,
        }],
        'auth_profile':
        'ambient',
    }
    source_spec = {'source': source, 'source_size': 123}
    fake_result = IngestResult(
        index_path=str(local_dir),
        col_stats_path=str(local_dir / 'col_stats.json'),
        ingest_stats_path=str(local_dir / 'ingest_stats.json'),
        kumo_stats_dir=str(local_dir / 'kumo_col_stats'),
    )

    direct_table = mock.MagicMock()
    staged_table = mock.MagicMock()
    staged_table.fingerprint.return_value = mock.sentinel.staged_fingerprint
    staged_source = str(tmp_path / 'staged-source')

    def ingest(*_args, **kwargs):
        assert kwargs['source_specs'] == {'direct': source_spec}
        assert kwargs['source_paths'] == {'staged': staged_source}
        assert 'databricks_config_file' not in kwargs
        local_dir.mkdir()
        (local_dir / 'snapshot.bin').write_bytes(b'payload')
        return fake_result

    ingest_args = SimpleNamespace(
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={
            'direct': 'v1',
            'staged': 'v2'
        },
        connector_specs={'s3': mock.sentinel.connector_ref},
        all_graph_tables={
            'direct': direct_table,
            'staged': staged_table,
        },
        source_size_bytes_by_table={
            'direct': 123,
            'staged': 456
        },
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.restore_snapshot.return_value = None
    fake_service.prepare_direct_sources.return_value = {'direct': source_spec}
    fake_service.prepare_subprocess_environment.return_value = {}
    fake_service.ingest.side_effect = ingest
    snapshot_ingest_dir = str(tmp_path / 'snapshot-download')
    download_cancellation_args = []
    ingest_cancellation_args = []

    async def _run_with_cleanup_capture(fn, *args, **kwargs):
        if fn is fake_service.download_table_files:
            download_cancellation_args.append((
                kwargs.get('join_on_cancel'),
                kwargs.get('cancellation_cleanup_dir'),
            ))
        if fn is fake_service.ingest:
            ingest_cancellation_args.append(kwargs.get('join_on_cancel'))
        return await _run_diskgraph_async(fn, *args, **kwargs)

    with (
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities.get_diskgraph_snapshot_ingest_path',
                       return_value=staged_source) as ingest_path,
            mock.patch('pool.activities.get_diskgraph_snapshot_ingest_dir',
                       return_value=snapshot_ingest_dir) as ingest_dir,
            mock.patch('pool.activities._build_diskgraph_ingest_config',
                       return_value=mock.sentinel.graph_config),
            mock.patch('pool.activities._run_diskgraph_async',
                       _run_with_cleanup_capture),
    ):
        result = await instance._ensure_snapshot_local(
            _TEST_SNAPSHOT_ID,
            ingest_args=ingest_args,
        )

    assert result is fake_result
    prepare_call = fake_service.prepare_direct_sources.call_args.kwargs
    assert prepare_call['data_persist_dir'] == ingest_args.data_persist_dir
    assert prepare_call['table_data_versions'] == (
        ingest_args.table_data_versions)
    ingest_path.assert_called_once_with(
        _TEST_SNAPSHOT_ID,
        mock.sentinel.staged_fingerprint,
        'v2',
    )
    ingest_dir.assert_called_once_with(_TEST_SNAPSHOT_ID)
    assert download_cancellation_args == [(True, snapshot_ingest_dir)]
    assert ingest_cancellation_args == [True]
    fake_service.download_table_files.assert_called_once_with(
        connector_specs=ingest_args.connector_specs,
        table_data_versions=ingest_args.table_data_versions,
        all_graph_tables=ingest_args.all_graph_tables,
        source_paths={'staged': staged_source},
    )
    fake_service.prepare_subprocess_environment.assert_called_once_with(
        connector_specs=ingest_args.connector_specs,
        all_graph_tables=ingest_args.all_graph_tables,
        source_specs={'direct': source_spec},
        data_persist_dir=ingest_args.data_persist_dir,
        table_data_versions=ingest_args.table_data_versions,
    )
    fake_service.ingest.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_ingest_holds_snapshot_lock_until_native_terminal(
        tmp_path):
    from types import SimpleNamespace

    from kumo.diskgraph import IngestResult

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    local_dir = tmp_path / 'cancelled-ingest'
    started = threading.Event()
    release = threading.Event()
    retry_ingest_started = threading.Event()
    retry_integrity_check = threading.Event()
    ingest_calls = 0
    ingest_calls_lock = threading.Lock()
    fake_result = IngestResult(
        index_path=str(local_dir),
        col_stats_path=str(local_dir / 'col_stats.json'),
        ingest_stats_path=str(local_dir / 'ingest_stats.json'),
        kumo_stats_dir=str(local_dir / 'kumo_col_stats'),
    )

    def ingest(*_args, **_kwargs):
        nonlocal ingest_calls
        with ingest_calls_lock:
            ingest_calls += 1
            call_number = ingest_calls
        if call_number == 1:
            started.set()
            assert release.wait(timeout=5)
        else:
            retry_ingest_started.set()
        local_dir.mkdir(exist_ok=True)
        (local_dir / 'snapshot.bin').write_text(f'attempt-{call_number}')
        return fake_result

    def verify_integrity(_handle):
        retry_integrity_check.set()
        return fake_result

    ingest_args = SimpleNamespace(
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={},
        connector_specs={},
        all_graph_tables={},
        source_size_bytes_by_table={},
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.restore_snapshot.return_value = None
    fake_service.has_resumable_ingest.return_value = False
    fake_service.has_persisted_ingest_config.return_value = False
    fake_service.prepare_direct_sources.return_value = {}
    fake_service.prepare_subprocess_environment.return_value = {}
    fake_service.verify_integrity.side_effect = verify_integrity
    fake_service.ingest.side_effect = ingest

    with (
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities._build_diskgraph_ingest_config',
                       return_value=mock.sentinel.graph_config),
    ):
        old_attempt = asyncio.create_task(
            instance._ensure_snapshot_local(
                _TEST_SNAPSHOT_ID,
                ingest_args=ingest_args,
            ))
        assert await asyncio.to_thread(started.wait, 1)
        old_attempt.cancel()
        retry = asyncio.create_task(
            instance._ensure_snapshot_local(
                _TEST_SNAPSHOT_ID,
                ingest_args=ingest_args,
            ))
        await asyncio.sleep(0.05)
        raced = (retry_ingest_started.is_set()
                 or retry_integrity_check.is_set() or retry.done())

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await old_attempt
        assert await asyncio.wait_for(retry, timeout=1) is fake_result

    assert not raced, 'retry entered before cancelled native ingest terminated'
    assert ingest_calls == 1
    assert retry_integrity_check.is_set()


@pytest.mark.asyncio
async def test_ensure_snapshot_local_no_space_removes_partial_attempt(tmp_path):
    from kumo.diskgraph.errors import DiskgraphNoSpaceError

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    instance._ready_snapshots.add(_TEST_SNAPSHOT_ID)
    local_dir = tmp_path / 'partial-local'
    dataplane_dir = tmp_path / 'partial-dataplane'

    def _ingest(*_args, **_kwargs):
        local_dir.mkdir()
        dataplane_dir.mkdir()
        (local_dir / 'partial.bin').write_bytes(b'partial')
        (dataplane_dir / 'partial.bin').write_bytes(b'partial')
        raise DiskgraphNoSpaceError('local SSD full')

    ingest_args = SimpleNamespace(
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={'orders': 'v1'},
        connector_specs={'dbx': mock.sentinel.connector_ref},
        all_graph_tables={'orders': mock.MagicMock()},
        source_size_bytes_by_table={'orders': 456},
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(dataplane_dir)
    fake_service.restore_snapshot.return_value = None
    fake_service.prepare_direct_sources.return_value = {
        'orders': mock.sentinel.source_spec
    }
    fake_service.ingest.side_effect = _ingest

    with (
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities._build_diskgraph_ingest_config',
                       return_value=mock.sentinel.graph_config),
    ):
        with pytest.raises(DiskgraphNoSpaceError):
            await instance._ensure_snapshot_local(
                _TEST_SNAPSHOT_ID,
                ingest_args=ingest_args,
            )

    assert not local_dir.exists()
    assert not dataplane_dir.exists()
    assert _TEST_SNAPSHOT_ID not in instance._ready_snapshots
    fake_service.cleanup_ingest_sidecars.assert_called()


@pytest.mark.asyncio
async def test_ensure_snapshot_local_resumes_after_encode_without_credentials(
        tmp_path):
    from types import SimpleNamespace

    from kumo.diskgraph import IngestResult

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    local_dir = tmp_path / 'incomplete'
    local_dir.mkdir()
    (local_dir / 'encode.bin').write_bytes(b'encoded')
    fake_result = IngestResult(
        index_path=str(local_dir),
        col_stats_path=str(local_dir / 'col_stats.json'),
        ingest_stats_path=str(local_dir / 'ingest_stats.json'),
        kumo_stats_dir=str(local_dir / 'kumo_col_stats'),
    )
    ingest_args = SimpleNamespace(
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={'orders': 'v1'},
        connector_specs={'dbx': mock.sentinel.connector_ref},
        all_graph_tables={'orders': mock.MagicMock()},
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.has_resumable_ingest.return_value = True
    fake_service.prepare_subprocess_environment.return_value = {}
    fake_service.ingest.return_value = fake_result

    with (
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
    ):
        result = await instance._ensure_snapshot_local(
            _TEST_SNAPSHOT_ID,
            ingest_args=ingest_args,
        )

    assert result is fake_result
    fake_service.verify_integrity.assert_not_called()
    fake_service.restore_snapshot.assert_not_called()
    fake_service.prepare_direct_sources.assert_not_called()
    fake_service.download_table_files.assert_not_called()
    assert fake_service.ingest.call_args.kwargs['graph_config'] is None
    assert fake_service.ingest.call_args.kwargs['source_specs'] is None
    fake_service.prepare_subprocess_environment.assert_called_once_with(
        connector_specs=ingest_args.connector_specs,
        all_graph_tables=ingest_args.all_graph_tables,
        source_specs=None,
        data_persist_dir=ingest_args.data_persist_dir,
        table_data_versions=ingest_args.table_data_versions,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('existing_partial_output', [True, False])
async def test_ensure_snapshot_local_early_retry_reuses_persisted_sources(
        tmp_path, existing_partial_output):
    from types import SimpleNamespace

    from kumo.diskgraph import IngestResult

    instance = _make_instance()
    instance._send_status_report = mock.AsyncMock()
    local_dir = tmp_path / 'incomplete'
    if existing_partial_output:
        local_dir.mkdir()
        (local_dir / 'partial.bin').write_bytes(b'partial')
    fake_result = IngestResult(
        index_path=str(local_dir),
        col_stats_path=str(local_dir / 'col_stats.json'),
        ingest_stats_path=str(local_dir / 'ingest_stats.json'),
        kumo_stats_dir=str(local_dir / 'kumo_col_stats'),
    )
    ingest_args = SimpleNamespace(
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={'events': 'v1'},
        connector_specs={'files': mock.sentinel.connector_ref},
        all_graph_tables={'events': mock.MagicMock()},
    )
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'missing-dataplane')
    fake_service.has_resumable_ingest.return_value = False
    fake_service.has_persisted_ingest_config.return_value = True
    fake_service.verify_integrity.side_effect = _DiskgraphCorruptionError(
        'interrupted before Encode completed')
    fake_service.restore_snapshot.return_value = None
    fake_service.prepare_subprocess_environment.return_value = {}

    def ingest(**_kwargs: object) -> IngestResult:
        local_dir.mkdir()
        (local_dir / 'index.bin').write_bytes(b'complete')
        return fake_result

    fake_service.ingest.side_effect = ingest

    with (
            mock.patch.dict(sys.modules, {'diskgraph': _FAKE_DISKGRAPH}),
            mock.patch('pool.activities.get_service',
                       return_value=fake_service),
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')),
            mock.patch('pool.activities._build_diskgraph_ingest_config') as
            build_config,
    ):
        result = await instance._ensure_snapshot_local(
            _TEST_SNAPSHOT_ID,
            ingest_args=ingest_args,
        )

    assert result is fake_result
    fake_service.prepare_direct_sources.assert_not_called()
    fake_service.download_table_files.assert_not_called()
    build_config.assert_not_called()
    ingest_kwargs = fake_service.ingest.call_args.kwargs
    assert ingest_kwargs['graph_config'] is None
    assert ingest_kwargs['source_specs'] is None
    assert ingest_kwargs['source_paths'] is None
    fake_service.prepare_subprocess_environment.assert_called_once_with(
        connector_specs=ingest_args.connector_specs,
        all_graph_tables=ingest_args.all_graph_tables,
        source_specs=None,
        data_persist_dir=ingest_args.data_persist_dir,
        table_data_versions=ingest_args.table_data_versions,
    )
    assert any(
        call.kwargs.get('preserve_config') is True
        for call in fake_service.cleanup_ingest_sidecars.call_args_list)


@pytest.mark.asyncio
async def test_ensure_snapshot_local_consumer_mode_raises_on_full_miss(
        tmp_path):
    """SSD miss + dataplane miss + ingest_args=None (consumer mode) →
    PERMANENT_SNAPSHOT_MISS_ERROR_TYPE non-retryable. The controller is
    expected to re-place via a fresh daemon dispatch; the consumer
    activity retries on the next dispatch."""
    from temporalio.exceptions import ApplicationError

    from pool.activities import PERMANENT_SNAPSHOT_MISS_ERROR_TYPE
    instance = _make_instance()

    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(tmp_path /
                                                        'nonexistent-local')
    fake_service.dataplane_snapshot_path.return_value = str(tmp_path /
                                                            'dataplane-mirror')
    fake_service.restore_snapshot.return_value = None  # dataplane miss

    with mock.patch('pool.activities.get_service',
                    return_value=fake_service), \
            mock.patch('pool.activities.get_persist_dir',
                       return_value=str(tmp_path / 'dataplane-root')):
        with pytest.raises(ApplicationError) as excinfo:
            await instance._ensure_snapshot_local(_TEST_SNAPSHOT_ID,
                                                  ingest_args=None)

    assert excinfo.value.non_retryable is True
    assert excinfo.value.type == PERMANENT_SNAPSHOT_MISS_ERROR_TYPE
    fake_service.ingest.assert_not_called()


@pytest.mark.asyncio
async def test_delete_snapshot_removes_local_snapshot_and_ready_state(tmp_path):
    instance = _make_instance()
    instance._ready_snapshots.add(_TEST_SNAPSHOT_ID)

    snapshot_parent = tmp_path / 'NA-graph-1bf8d6a669f6a6caba68a2393f1044e3'
    local_dir = snapshot_parent / 'v1'
    local_dir.mkdir(parents=True)
    (local_dir / '_SUCCESS').write_text('')
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    instance._send_status_report = mock.AsyncMock()

    with mock.patch('pool.activities.get_service', return_value=fake_service):
        await instance.delete_snapshot(_TEST_SNAPSHOT_ID)

    assert not local_dir.exists()
    assert snapshot_parent.exists()
    assert _TEST_SNAPSHOT_ID not in instance._ready_snapshots
    instance._send_status_report.assert_awaited_once_with(raise_on_error=False)


@pytest.mark.asyncio
async def test_delete_snapshot_is_idempotent_when_local_snapshot_missing(
        tmp_path):
    instance = _make_instance()
    local_dir = tmp_path / 'missing'
    fake_service = mock.MagicMock()
    fake_service.local_snapshot_path.return_value = str(local_dir)
    instance._send_status_report = mock.AsyncMock()

    with mock.patch('pool.activities.get_service', return_value=fake_service):
        await instance.delete_snapshot(_TEST_SNAPSHOT_ID)

    assert not local_dir.exists()
    instance._send_status_report.assert_awaited_once_with(raise_on_error=False)


# ----------------------------------------------------------------------
# Sharded prediction: _shard_pred_config
# ----------------------------------------------------------------------


def _make_pred_config(num_partitions: int, bundle_dir: str | None = None):
    from kumo_core.job_config import PredictionConfig
    return PredictionConfig(
        home_dir='/tmp/h',
        train_dir='/tmp/t',
        task_name='task',
        model_dir='/tmp/m',
        output_dir='/tmp/o',
        return_embeddings=False,
        return_predictions=True,
        pred_table_path='/tmp/p',
        num_partitions=num_partitions,
        start_partition_id=0,
        stop_partition_id=1,
        diskgraph_task_dataset_bundle_dir=bundle_dir,
    )


def test_shard_pred_config_no_op_when_num_shards_le_one(tmp_path):
    orig_path = str(tmp_path / 'pred_config.json')
    # File needn't exist — short-circuits before any I/O.
    assert _shard_pred_config(orig_path, 0, num_shards=1) == orig_path
    assert _shard_pred_config(orig_path, 0, num_shards=0) == orig_path


def test_shard_pred_config_writes_sharded_file(tmp_path):
    from kumo_core.job_config import PredictionConfig

    from io_utils.json import load_json, save_json

    orig_path = str(tmp_path / 'pred_config.json')
    save_json(_make_pred_config(num_partitions=4), orig_path)

    p0 = _shard_pred_config(orig_path, 0, num_shards=2)
    p1 = _shard_pred_config(orig_path, 1, num_shards=2)

    assert len({p0, p1, orig_path}) == 3, 'sharded paths must be distinct'
    s0 = load_json(p0, PredictionConfig)
    s1 = load_json(p1, PredictionConfig)
    assert (s0.start_partition_id, s0.stop_partition_id) == (0, 2)
    assert (s1.start_partition_id, s1.stop_partition_id) == (2, 4)


def test_shard_pred_config_can_redirect_output_to_worker_nvme(tmp_path):
    from kumo_core.job_config import PredictionConfig

    from io_utils.json import load_json, save_json

    orig_path = str(tmp_path / 'pred_config.json')
    save_json(_make_pred_config(num_partitions=1), orig_path)
    local_output = str(tmp_path / 'worker-output')

    sharded_path = _shard_pred_config(
        orig_path,
        0,
        num_shards=1,
        output_dir=local_output,
    )

    assert sharded_path != orig_path
    sharded = load_json(sharded_path, PredictionConfig)
    assert sharded.output_dir == local_output
    assert sharded.max_prediction_file_size_bytes == 128 << 20
    assert load_json(orig_path, PredictionConfig).output_dir == '/tmp/o'


def test_shard_pred_config_attempts_cannot_share_config_or_output(tmp_path):
    from kumo_core.job_config import PredictionConfig

    from io_utils.json import load_json, save_json

    orig_path = str(tmp_path / 'pred_config.json')
    save_json(_make_pred_config(num_partitions=2), orig_path)
    first_output = str(tmp_path / '_attempts' / 'first')
    second_output = str(tmp_path / '_attempts' / 'second')

    first = _shard_pred_config(
        orig_path,
        0,
        num_shards=2,
        output_dir=first_output,
        attempt_id='first',
    )
    second = _shard_pred_config(
        orig_path,
        0,
        num_shards=2,
        output_dir=second_output,
        attempt_id='second',
    )

    assert first != second
    assert load_json(first, PredictionConfig).output_dir == first_output
    assert load_json(second, PredictionConfig).output_dir == second_output


@pytest.mark.parametrize('num_shards,stage_suffix,producer_abi', [
    (1, '', 'kumo-pred-output-v2'),
    (2, ':kumo-pred-output-v3', 'kumo-pred-output-v3'),
])
def test_databricks_upload_uses_immutable_attempt_prefix(
    tmp_path,
    monkeypatch,
    num_shards,
    stage_suffix,
    producer_abi,
):
    from kumo_core.types import PredictionArtifactType

    import kumo.connector
    import rest.utils.driver_common

    artifact_dir = tmp_path / 'predictions.parquet'
    artifact_dir.mkdir()
    part = artifact_dir / 'part-00000-00000.parquet'
    part.write_bytes(b'part')
    calls = []

    class Connector:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def ensure_managed_volume(self):
            return None

        def managed_volume_stage(self, name):
            calls.append(('stage', name))
            return '/Volumes/catalog/schema/volume/job'

        def upload_local_artifacts_to_volume(
            self,
            path,
            *,
            stage,
            file_prefix,
            files,
        ):
            calls.append(('upload', path, stage, file_prefix, files))
            return stage, [f'{stage}/{file_prefix}-00000.parquet']

    connector = Connector()
    monkeypatch.setattr(kumo.connector, 'DatabricksConnector', Connector)
    monkeypatch.setattr(rest.utils.driver_common, 'get_connector_from_spec',
                        lambda _spec: connector)
    monkeypatch.setattr(
        activities.parquet_util,
        'get_parquet_metadata',
        lambda _path: SimpleNamespace(num_rows=3, num_cols=2, avg_row_size=4),
    )
    monkeypatch.setattr(activities, 'local_parquet_files', lambda _path: [part])

    result = _upload_prediction_artifact_to_databricks(
        object(),
        artifact_path=str(artifact_dir),
        artifact_type=PredictionArtifactType.PREDICTIONS,
        job_id='bp-job-1',
        output_index=0,
        shard_index=3,
        attempt_id='attempt-a',
        num_shards=num_shards,
    )

    # Older finalizers see mixed producers as mixed stage paths.
    assert calls[0] == ('stage', 'batch-prediction:bp-job-1:0:'
                        f'{PredictionArtifactType.PREDICTIONS.value}'
                        f'{stage_suffix}')
    assert result.producer_abi == producer_abi
    assert calls[1][3] == 'shard-00003-attempt-attempt-a'
    assert result.staged_files == [
        '/Volumes/catalog/schema/volume/job/'
        'shard-00003-attempt-attempt-a-00000.parquet'
    ]


def test_shard_pred_config_rejects_more_shards_than_partitions(tmp_path):
    from io_utils.json import save_json

    orig_path = str(tmp_path / 'pred_config.json')
    save_json(_make_pred_config(num_partitions=2), orig_path)

    with pytest.raises(ValueError, match='num_partitions=2'):
        _shard_pred_config(orig_path, 0, num_shards=4)


def test_shard_pred_config_assigns_ttng_worker_shards(tmp_path):
    from kumo_core.job_config import PredictionConfig

    from io_utils.json import load_json, save_json

    orig_path = str(tmp_path / 'pred_config.json')
    save_json(_make_pred_config(num_partitions=4, bundle_dir='/tmp/bundle'),
              orig_path)

    p0 = _shard_pred_config(orig_path, 0, num_shards=4)
    p3 = _shard_pred_config(orig_path, 3, num_shards=4)

    s0 = load_json(p0, PredictionConfig)
    s3 = load_json(p3, PredictionConfig)
    assert (s0.start_partition_id, s0.stop_partition_id) == (0, 1)
    assert (s3.start_partition_id, s3.stop_partition_id) == (3, 4)
    assert s0.num_partitions == 4
    assert s3.num_partitions == 4


def test_ttng_output_partition_ranges_follow_kumo_pred_contract():
    from kumo.train.prediction_partitions import (
        prediction_output_partition_count,
        prediction_output_partition_range,
    )

    config = _make_pred_config(num_partitions=4, bundle_dir='/tmp/bundle')
    config.output_num_partitions = 97
    config.start_partition_id = 2
    config.stop_partition_id = 3

    assert prediction_output_partition_range(config) == (49, 73)
    assert prediction_output_partition_count(config) == 97


def test_legacy_output_partition_range_remains_direct():
    from kumo.train.prediction_partitions import (
        prediction_output_partition_count,
        prediction_output_partition_range,
    )

    config = _make_pred_config(num_partitions=7)
    config.start_partition_id = 2
    config.stop_partition_id = 5
    config.output_num_partitions = 97

    assert prediction_output_partition_range(config) == (2, 5)
    assert prediction_output_partition_count(config) == 7


def _write_runtime_manifest(bundle_dir, stage_index_path, stage_name='predict'):
    from io_utils.json import save_json

    bundle_dir.mkdir(parents=True, exist_ok=True)
    save_json({'metadata_path': 'metadata.json'},
              str(bundle_dir / 'manifest.json'))
    save_json(
        {
            'stages': {
                stage_name: {
                    'index_path': str(stage_index_path),
                },
            },
        },
        str(bundle_dir / 'runtime_manifest.json'),
    )


def test_ttng_job_sidecars_upload_restore_and_cleanup_prediction(tmp_path):
    import shutil

    from io_utils.json import load_json, save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'bp-job'
    extension = index / 'pred_tables' / 'bp-job'
    for path, marker in [(bundle, 'bundle'), (extension, 'extension')]:
        path.mkdir(parents=True)
        (path / 'marker').write_text(marker)
    _write_runtime_manifest(bundle, extension)

    config = _make_pred_config(num_partitions=1, bundle_dir=str(bundle))
    config.diskgraph_index_path = str(index)
    config_path = tmp_path / 'pred_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    remote_root = upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )

    assert remote_root is not None
    manifest = load_json(str(Path(remote_root) / 'manifests' / 'bp-job.json'),
                         dict)
    generation_root = Path(remote_root) / manifest['generation_relpath']
    assert (generation_root / 'task_datasets' / 'bp-job' /
            'marker').read_text() == 'bundle'
    assert (generation_root / 'pred_tables' / 'bp-job' /
            'marker').read_text() == 'extension'
    assert manifest['schema_version'] == 2
    assert manifest['relpaths'] == [
        'pred_tables/bp-job',
        'task_datasets/bp-job',
    ]

    shutil.rmtree(index / 'task_datasets')
    shutil.rmtree(index / 'pred_tables')
    restore_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    assert (bundle / 'marker').read_text() == 'bundle'
    assert (extension / 'marker').read_text() == 'extension'

    cleanup_local_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    assert not bundle.exists()
    assert not extension.exists()


def test_ttng_sidecar_retry_restores_exact_winning_generation(
        tmp_path, monkeypatch):
    import shutil

    from io_utils.json import load_json, save_json
    from kumo.diskgraph import job_sidecars as job_sidecars_mod

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'bp-job'
    extension = index / 'train_tables' / 'pred_bp-job'
    _write_runtime_manifest(bundle, extension)
    extension.mkdir(parents=True)
    (extension / 'diskgraph_manifest.json').write_text('{}')
    for root in (bundle, extension):
        (root / 'keep').write_text('old')
        (root / 'stale').write_text('stale')

    config = _make_pred_config(num_partitions=1, bundle_dir=str(bundle))
    config.diskgraph_index_path = str(index)
    config_path = tmp_path / 'pred_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'
    remote_root = upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    manifest_path = Path(remote_root) / 'manifests' / 'bp-job.json'
    old_manifest = load_json(str(manifest_path), dict)

    for root in (bundle, extension):
        (root / 'stale').unlink()
        (root / 'keep').write_text('winning')

    original_copy = job_sidecars_mod._copy_relpath
    copied = 0

    def fail_after_first_copy(src_root, dst_root, relpath):
        nonlocal copied
        original_copy(src_root, dst_root, relpath)
        copied += 1
        if copied == 1:
            raise RuntimeError('injected partial generation upload')

    monkeypatch.setattr(job_sidecars_mod, '_copy_relpath',
                        fail_after_first_copy)
    with pytest.raises(RuntimeError, match='injected partial'):
        upload_job_sidecars_from_config_path(
            _TEST_SNAPSHOT_ID,
            str(config_path),
            prediction=True,
            data_persist_dir=str(persist),
        )
    assert load_json(str(manifest_path), dict) == old_manifest

    monkeypatch.setattr(job_sidecars_mod, '_copy_relpath', original_copy)
    upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    winning_manifest = load_json(str(manifest_path), dict)
    assert (winning_manifest['generation_relpath']
            != old_manifest['generation_relpath'])

    shutil.rmtree(index / 'task_datasets')
    shutil.rmtree(index / 'train_tables')
    restore_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    assert sorted(
        path.relative_to(bundle).as_posix() for path in bundle.rglob('*')
        if path.is_file()) == [
            'keep', 'manifest.json', 'runtime_manifest.json'
        ]
    assert sorted(
        path.relative_to(extension).as_posix() for path in extension.rglob('*')
        if path.is_file()) == ['diskgraph_manifest.json', 'keep']
    assert (bundle / 'keep').read_text() == 'winning'
    assert (extension / 'keep').read_text() == 'winning'


def test_ttng_manifest_pointer_accepts_lost_success_ack(tmp_path, monkeypatch):
    from io_utils import fsutil
    from io_utils.json import load_json
    from kumo.diskgraph import job_sidecars as job_sidecars_mod

    manifest_path = str(tmp_path / 'manifest.json')
    manifest = {
        'schema_version': 2,
        'snapshot_id': _TEST_SNAPSHOT_ID,
        'job_id': 'bp-job',
        'generation_relpath': 'generations/bp-job/generation-1',
    }
    original_copy = fsutil.copy

    def copy_then_lose_ack(*args, **kwargs):
        original_copy(*args, **kwargs)
        raise RuntimeError('lost remote acknowledgement')

    monkeypatch.setattr(fsutil, 'is_local', lambda _path: False)
    monkeypatch.setattr(fsutil, 'copy', copy_then_lose_ack)

    job_sidecars_mod._publish_manifest_pointer(manifest, manifest_path)
    job_sidecars_mod._publish_manifest_pointer(manifest, manifest_path)

    assert load_json(manifest_path, dict) == manifest
    assert not list(tmp_path.glob('.manifest.json.attempt-*'))


def test_ttng_manifest_pointer_does_not_swallow_base_exception(
        tmp_path, monkeypatch):
    from io_utils import fsutil
    from io_utils.json import load_json
    from kumo.diskgraph import job_sidecars as job_sidecars_mod

    class ProcessControlSignal(BaseException):
        pass

    manifest_path = str(tmp_path / 'manifest.json')
    manifest = {
        'schema_version': 2,
        'snapshot_id': _TEST_SNAPSHOT_ID,
        'job_id': 'bp-job',
        'generation_relpath': 'generations/bp-job/generation-1',
    }
    original_copy = fsutil.copy

    def copy_then_interrupt(*args, **kwargs):
        original_copy(*args, **kwargs)
        raise ProcessControlSignal('stop process')

    monkeypatch.setattr(fsutil, 'is_local', lambda _path: False)
    monkeypatch.setattr(fsutil, 'copy', copy_then_interrupt)

    with pytest.raises(ProcessControlSignal, match='stop process'):
        job_sidecars_mod._publish_manifest_pointer(manifest, manifest_path)

    assert load_json(manifest_path, dict) == manifest
    assert not list(tmp_path.glob('.manifest.json.attempt-*'))


def test_ttng_manifest_pointer_pre_put_failure_preserves_old_pointer(
        tmp_path, monkeypatch):
    from io_utils import fsutil
    from io_utils.json import load_json, save_json
    from kumo.diskgraph import job_sidecars as job_sidecars_mod

    manifest_path = str(tmp_path / 'manifest.json')
    old_manifest = {
        'schema_version': 2,
        'snapshot_id': _TEST_SNAPSHOT_ID,
        'job_id': 'bp-job',
        'generation_relpath': 'generations/bp-job/old',
    }
    new_manifest = {
        **old_manifest,
        'generation_relpath': 'generations/bp-job/new',
    }
    save_json(old_manifest, manifest_path)
    monkeypatch.setattr(fsutil, 'is_local', lambda _path: False)
    monkeypatch.setattr(
        fsutil,
        'copy',
        mock.MagicMock(side_effect=RuntimeError('put failed before commit')),
    )

    with pytest.raises(RuntimeError, match='before commit'):
        job_sidecars_mod._publish_manifest_pointer(new_manifest, manifest_path)

    assert load_json(manifest_path, dict) == old_manifest
    assert not list(tmp_path.glob('.manifest.json.attempt-*'))


@pytest.mark.parametrize('final_pointer', ['malformed', 'different'])
def test_ttng_manifest_pointer_rejects_nonmatching_post_error_readback(
        tmp_path, monkeypatch, final_pointer):
    from io_utils import fsutil
    from io_utils.json import load_json, save_json
    from kumo.diskgraph import job_sidecars as job_sidecars_mod

    manifest_path = str(tmp_path / 'manifest.json')
    intended = {
        'schema_version': 2,
        'snapshot_id': _TEST_SNAPSHOT_ID,
        'job_id': 'bp-job',
        'generation_relpath': 'generations/bp-job/intended',
    }
    different = {
        **intended,
        'generation_relpath': 'generations/bp-job/different',
    }

    def put_other_then_raise(_src, dst, **_kwargs):
        if final_pointer == 'malformed':
            Path(dst).write_text('{broken')
        else:
            save_json(different, dst)
        raise RuntimeError('ambiguous put result')

    monkeypatch.setattr(fsutil, 'is_local', lambda _path: False)
    monkeypatch.setattr(fsutil, 'copy', put_other_then_raise)

    with pytest.raises(RuntimeError, match='ambiguous put result'):
        job_sidecars_mod._publish_manifest_pointer(intended, manifest_path)

    if final_pointer == 'different':
        assert load_json(manifest_path, dict) == different
    assert not list(tmp_path.glob('.manifest.json.attempt-*'))


def test_ttng_upload_lost_pointer_ack_keeps_selected_generation(
        tmp_path, monkeypatch):
    import shutil

    from io_utils import fsutil
    from io_utils.json import load_json, save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'bp-job'
    extension = index / 'train_tables' / 'pred_bp-job'
    _write_runtime_manifest(bundle, extension)
    extension.mkdir(parents=True)
    (extension / 'diskgraph_manifest.json').write_text('{}')
    (bundle / 'marker').write_text('bundle')
    (extension / 'marker').write_text('extension')
    config = _make_pred_config(num_partitions=1, bundle_dir=str(bundle))
    config.diskgraph_index_path = str(index)
    config_path = tmp_path / 'pred_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'
    manifest_path = (persist / 'diskgraph_job_sidecars' /
                     'NA-1bf8d6a669f6a6caba68a2393f1044e3' / 'v1' /
                     'manifests' / 'bp-job.json')
    original_copy = fsutil.copy

    def copy_then_lose_pointer_ack(src, dst, **kwargs):
        original_copy(src, dst, **kwargs)
        if str(dst) == str(manifest_path):
            raise RuntimeError('lost pointer acknowledgement')

    monkeypatch.setattr(fsutil, 'is_local', lambda _path: False)
    monkeypatch.setattr(fsutil, 'copy', copy_then_lose_pointer_ack)

    remote_root = upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    manifest = load_json(str(manifest_path), dict)
    generation_root = Path(remote_root) / manifest['generation_relpath']
    assert generation_root.is_dir()

    shutil.rmtree(index / 'task_datasets')
    shutil.rmtree(index / 'train_tables')
    restore_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    assert (bundle / 'marker').read_text() == 'bundle'
    assert (extension / 'marker').read_text() == 'extension'


def test_ttng_upload_pre_put_failure_cleans_generation_and_retries(
        tmp_path, monkeypatch):
    from io_utils import fsutil
    from io_utils.json import load_json, save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'bp-job'
    extension = index / 'train_tables' / 'pred_bp-job'
    _write_runtime_manifest(bundle, extension)
    extension.mkdir(parents=True)
    (extension / 'diskgraph_manifest.json').write_text('{}')
    (bundle / 'marker').write_text('old')
    (extension / 'marker').write_text('old')
    config = _make_pred_config(num_partitions=1, bundle_dir=str(bundle))
    config.diskgraph_index_path = str(index)
    config_path = tmp_path / 'pred_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'
    remote_root = upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    manifest_path = Path(remote_root) / 'manifests' / 'bp-job.json'
    old_manifest = load_json(str(manifest_path), dict)
    generations_dir = Path(remote_root) / 'generations' / 'bp-job'
    old_generations = sorted(path.name for path in generations_dir.iterdir())
    (bundle / 'marker').write_text('new')
    (extension / 'marker').write_text('new')
    original_copy = fsutil.copy

    def fail_pointer_put(src, dst, **kwargs):
        if str(dst) == str(manifest_path):
            raise RuntimeError('pointer pre-put failure')
        return original_copy(src, dst, **kwargs)

    monkeypatch.setattr(fsutil, 'is_local', lambda _path: False)
    monkeypatch.setattr(fsutil, 'copy', fail_pointer_put)
    with pytest.raises(RuntimeError, match='pointer pre-put failure'):
        upload_job_sidecars_from_config_path(
            _TEST_SNAPSHOT_ID,
            str(config_path),
            prediction=True,
            data_persist_dir=str(persist),
        )
    assert load_json(str(manifest_path), dict) == old_manifest
    assert sorted(path.name
                  for path in generations_dir.iterdir()) == old_generations

    monkeypatch.setattr(fsutil, 'copy', original_copy)
    upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    winning_manifest = load_json(str(manifest_path), dict)
    assert winning_manifest != old_manifest
    assert (Path(remote_root) / winning_manifest['generation_relpath'] /
            'task_datasets' / 'bp-job' / 'marker').read_text() == 'new'


def test_ttng_prediction_shared_sidecars_survive_active_shard(tmp_path):
    from io_utils.json import save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'bp-job'
    extension = index / 'train_tables' / 'pred_bp-job'
    for path, marker in [(bundle, 'bundle'), (extension, 'extension')]:
        path.mkdir(parents=True)
        (path / 'marker').write_text(marker)
    (extension / 'diskgraph_manifest.json').write_text('{}')
    _write_runtime_manifest(bundle, extension)

    config = _make_pred_config(num_partitions=4, bundle_dir=str(bundle))
    config.diskgraph_index_path = str(index)
    config_path = tmp_path / 'pred_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    assert upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    ) is not None

    shard_0 = restore_job_sidecars_for_use_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    shard_1 = restore_job_sidecars_for_use_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(persist),
    )
    assert shard_0 is not None
    assert shard_1 is not None

    shard_0.cleanup()
    assert bundle.exists()
    assert extension.exists()

    shard_1.cleanup()
    assert not bundle.exists()
    assert not extension.exists()


def test_ttng_job_sidecars_keep_flag_skips_cleanup(tmp_path, monkeypatch):
    from io_utils.json import save_json

    bundle = tmp_path / 'index' / 'task_datasets' / 'bp-job'
    extension = tmp_path / 'index' / 'pred_tables' / 'bp-job'
    for path in [bundle, extension]:
        path.mkdir(parents=True)
        (path / 'marker').write_text('x')
    _write_runtime_manifest(bundle, extension, stage_name='train')

    config = _make_pred_config(num_partitions=1, bundle_dir=str(bundle))
    config.diskgraph_index_path = str(tmp_path / 'index')
    config_path = tmp_path / 'pred_config.json'
    save_json(config, str(config_path))
    monkeypatch.setenv('KUMO_TTNG_KEEP_JOB_SIDECARS', 'true')

    cleanup_local_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=True,
        data_persist_dir=str(tmp_path / 'persist'),
    )

    assert bundle.exists()
    assert extension.exists()


def test_ttng_job_sidecars_include_training_extension(tmp_path):
    from kumo_core.job_config import JobConfig

    from io_utils.json import load_json, save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'train-job'
    extension = index / 'train_tables' / 'tt-job'
    for path, marker in [(bundle, 'bundle'), (extension, 'extension')]:
        path.mkdir(parents=True)
        (path / 'marker').write_text(marker)
    _write_runtime_manifest(bundle, extension)

    config = JobConfig(
        home_dir='/tmp/home',
        task_name='task',
        diskgraph_index_path=str(index),
        diskgraph_task_dataset_bundle_dir=str(bundle),
    )
    config_path = tmp_path / 'job_config.json'
    save_json(config, str(config_path))

    remote_root = upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(tmp_path / 'persist'),
    )

    assert remote_root is not None
    manifest = load_json(
        str(Path(remote_root) / 'manifests' / 'train-job.json'), dict)
    generation_root = Path(remote_root) / manifest['generation_relpath']
    assert (generation_root / 'task_datasets' / 'train-job' /
            'marker').read_text() == 'bundle'
    assert (generation_root / 'train_tables' / 'tt-job' /
            'marker').read_text() == 'extension'


def test_ttng_shared_train_table_restore_skips_complete_local_extension(
        tmp_path):
    from kumo_core.job_config import JobConfig

    from io_utils.json import save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'train-job'
    extension = index / 'train_tables' / 'tt-job'
    for path in [bundle, extension]:
        path.mkdir(parents=True)
    (bundle / 'marker').write_text('local-bundle')
    (extension / 'marker').write_text('local-extension')
    (extension / 'diskgraph_manifest.json').write_text('{}')
    _write_runtime_manifest(bundle, extension)

    config = JobConfig(
        home_dir='/tmp/home',
        task_name='task',
        diskgraph_index_path=str(index),
        diskgraph_task_dataset_bundle_dir=str(bundle),
    )
    config_path = tmp_path / 'job_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    remote_root = upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )
    assert remote_root is not None
    manifest = load_json(
        str(Path(remote_root) / 'manifests' / 'train-job.json'), dict)
    generation_root = Path(remote_root) / manifest['generation_relpath']
    (generation_root / 'task_datasets' / 'train-job' /
     'marker').write_text('remote-bundle')
    (generation_root / 'train_tables' / 'tt-job' /
     'marker').write_text('remote-extension')

    restore_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )

    assert (bundle / 'marker').read_text() == 'remote-bundle'
    assert (extension / 'marker').read_text() == 'local-extension'


def test_ttng_shared_train_table_use_restore_skips_exclusive_when_complete(
        tmp_path, monkeypatch):
    from kumo_core.job_config import JobConfig

    from io_utils.json import save_json
    from kumo.diskgraph import job_sidecars as job_sidecars_mod

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'train-job'
    extension = index / 'train_tables' / 'tt-job'
    for path in [bundle, extension]:
        path.mkdir(parents=True)
    (bundle / 'marker').write_text('local-bundle')
    (extension / 'marker').write_text('local-extension')
    (extension / 'diskgraph_manifest.json').write_text('{}')
    _write_runtime_manifest(bundle, extension)

    config = JobConfig(
        home_dir='/tmp/home',
        task_name='task',
        diskgraph_index_path=str(index),
        diskgraph_task_dataset_bundle_dir=str(bundle),
    )
    config_path = tmp_path / 'job_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    assert upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    ) is not None

    class SharedOnlyLock:
        def __init__(self, index_path, relpath):
            self.index_path = index_path
            self.relpath = relpath
            self.released = False

        def acquire_exclusive(self, *, blocking=True):
            raise AssertionError('complete shared restore took exclusive lock')

        def acquire_shared(self):
            pass

        def release(self):
            self.released = True

    monkeypatch.setattr(job_sidecars_mod, '_SharedSidecarLock', SharedOnlyLock)

    sidecars = restore_job_sidecars_for_use_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )
    try:
        assert sidecars is not None
        assert (bundle / 'marker').read_text() == 'local-bundle'
        assert (extension / 'marker').read_text() == 'local-extension'
    finally:
        sidecars.release()


def test_ttng_shared_train_table_use_restore_accepts_train_table_sentinel(
        tmp_path):
    from kumo_core.job_config import JobConfig

    from io_utils.json import save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'train-job'
    extension = index / 'train_tables' / 'tt-job'
    for path in [bundle, extension]:
        path.mkdir(parents=True)
    (bundle / 'marker').write_text('local-bundle')
    (extension / 'marker').write_text('local-extension')
    (extension / 'diskgraph_manifest.json').write_text('{}')
    _write_runtime_manifest(bundle, extension)

    config = JobConfig(
        home_dir='/tmp/home',
        task_name='task',
        diskgraph_index_path=str(index),
        diskgraph_task_dataset_bundle_dir=str(bundle),
    )
    config_path = tmp_path / 'job_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    assert upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    ) is not None

    shutil.rmtree(bundle)
    shutil.rmtree(extension)

    sidecars = restore_job_sidecars_for_use_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )
    try:
        assert sidecars is not None
        assert (bundle / 'marker').read_text() == 'local-bundle'
        assert (extension / 'marker').read_text() == 'local-extension'
        assert (extension / 'diskgraph_manifest.json').is_file()
    finally:
        sidecars.cleanup()


def test_ttng_shared_train_table_cleanup_skips_active_extension(tmp_path):
    from kumo_core.job_config import JobConfig

    from io_utils.json import save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'train-job'
    extension = index / 'train_tables' / 'tt-job'
    for path in [bundle, extension]:
        path.mkdir(parents=True)
    (bundle / 'marker').write_text('bundle')
    (extension / 'marker').write_text('extension')
    (extension / 'diskgraph_manifest.json').write_text('{}')
    _write_runtime_manifest(bundle, extension)

    config = JobConfig(
        home_dir='/tmp/home',
        task_name='task',
        diskgraph_index_path=str(index),
        diskgraph_task_dataset_bundle_dir=str(bundle),
    )
    config_path = tmp_path / 'job_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    assert upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    ) is not None

    sidecars = restore_job_sidecars_for_use_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )
    try:
        cleanup_local_job_sidecars_from_config_path(
            _TEST_SNAPSHOT_ID,
            str(config_path),
            prediction=False,
            data_persist_dir=str(persist),
        )

        assert not bundle.exists()
        assert extension.exists()
    finally:
        sidecars.release()

    cleanup_local_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )
    assert not extension.exists()


def test_ttng_shared_train_table_runtime_cleanup_releases_before_delete(
        tmp_path, monkeypatch):
    from kumo_core.job_config import JobConfig

    from io_utils.json import save_json

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'train-job'
    extension = index / 'train_tables' / 'tt-job'
    for path in [bundle, extension]:
        path.mkdir(parents=True)
    (bundle / 'marker').write_text('bundle')
    (extension / 'marker').write_text('extension')
    (extension / 'diskgraph_manifest.json').write_text('{}')
    _write_runtime_manifest(bundle, extension)

    config = JobConfig(
        home_dir='/tmp/home',
        task_name='task',
        diskgraph_index_path=str(index),
        diskgraph_task_dataset_bundle_dir=str(bundle),
    )
    config_path = tmp_path / 'job_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    assert upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    ) is not None

    sidecars = restore_job_sidecars_for_use_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )
    assert sidecars is not None
    held_lock, _relpath = sidecars._shared_locks[0]

    def fail_if_shared_lock_is_upgraded(*_args, **_kwargs):
        raise AssertionError('cleanup upgraded a held shared lock')

    monkeypatch.setattr(held_lock, 'acquire_exclusive',
                        fail_if_shared_lock_is_upgraded)

    sidecars.cleanup()

    assert not bundle.exists()
    assert not extension.exists()


def test_ttng_runtime_sidecar_cleanup_is_best_effort(tmp_path, monkeypatch):
    from kumo_core.job_config import JobConfig

    from io_utils.json import save_json
    from kumo.diskgraph import job_sidecars as job_sidecars_mod

    index = tmp_path / 'index'
    bundle = index / 'task_datasets' / 'train-job'
    extension = index / 'train_tables' / 'tt-job'
    for path in [bundle, extension]:
        path.mkdir(parents=True)
    (extension / 'diskgraph_manifest.json').write_text('{}')
    _write_runtime_manifest(bundle, extension)

    config = JobConfig(
        home_dir='/tmp/home',
        task_name='task',
        diskgraph_index_path=str(index),
        diskgraph_task_dataset_bundle_dir=str(bundle),
    )
    config_path = tmp_path / 'job_config.json'
    save_json(config, str(config_path))
    persist = tmp_path / 'persist'

    assert upload_job_sidecars_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    ) is not None

    sidecars = restore_job_sidecars_for_use_from_config_path(
        _TEST_SNAPSHOT_ID,
        str(config_path),
        prediction=False,
        data_persist_dir=str(persist),
    )
    assert sidecars is not None
    held_lock, _relpath = sidecars._shared_locks[0]
    release = mock.Mock(side_effect=RuntimeError('release failed'))
    monkeypatch.setattr(held_lock, 'release', release)

    cleanup = mock.Mock(side_effect=[
        RuntimeError('owned cleanup failed'),
        RuntimeError('shared cleanup failed'),
    ])
    monkeypatch.setattr(job_sidecars_mod, '_cleanup_local_job_sidecars',
                        cleanup)

    sidecars.cleanup()

    assert cleanup.call_count == 2
    release.assert_called_once_with()


def test_home_stale_error_stays_retryable_through_placement_chain():
    """STALE-wait foundation: a STALE home makes ``route()`` raise
    ``HomeStaleError``, and the whole wait depends on that error staying
    RETRYABLE end-to-end so Temporal keeps retrying ``place_snapshot_activity``
    until the home settles (``run_pooled_activity`` does NOT catch it). Lock
    the three invariants a refactor could silently break — if any breaks,
    every STALE event becomes a workflow failure instead of a transparent
    wait.
    """
    from temporalio.exceptions import ApplicationError

    from pool.activities import _raise_as_application_error
    from pool.errors import HomeStaleError
    from temporalio_commons.auto_rpc_exception import RpcError, is_retryable

    # 1. HomeStaleError extends RuntimeError — the branch is_retryable keys on.
    assert issubclass(HomeStaleError, RuntimeError)

    # 2. AutoRPC classifies it retryable → client RpcError(non_retryable=False).
    assert is_retryable(HomeStaleError('home is stale')) is True

    # 3. place_snapshot_activity translates that retryable RpcError into a
    #    RETRYABLE ApplicationError (non_retryable=False), so Temporal retries
    #    the placement (which IS the STALE wait) instead of failing.
    rpc_err = RpcError(
        rpc_service='PoolController',
        rpc_method='place_snapshot',
        error_type='HomeStaleError',
        message='home is stale',
        stack_trace='',
        non_retryable=False,
    )
    with pytest.raises(ApplicationError) as excinfo:
        _raise_as_application_error(rpc_err)
    assert excinfo.value.non_retryable is False
    assert excinfo.value.type == 'HomeStaleError'


def test_existing_worker_unavailable_error_stops_placement_retries():
    from pool.errors import ExistingWorkerUnavailableError
    from temporalio_commons.auto_rpc_exception import is_retryable

    error = ExistingWorkerUnavailableError('worker resources disappeared')

    assert not isinstance(error, RuntimeError)
    assert is_retryable(error) is False
