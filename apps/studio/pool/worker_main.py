# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool activity-worker entrypoint (``python -m pool.worker_main``).

Configured entirely via env vars (see ``EnvVar`` declarations below); the
spawner (k8s pod spec / singlehost bootstrap / test fixture) populates them.
"""
import asyncio
import contextlib
import dataclasses
import faulthandler
import inspect
import os
import resource
import signal
import sys
from datetime import timedelta
from typing import Any, Callable, List, Optional

from temporalio import activity
from temporalio.worker import Worker

from async_utils.loop_guards import install_loop_guards
from io_utils import logging
from pool.activities import SnapshotActivities
from pool.bootstrap import init_logging_and_tracing
from pool.client import get_controller_client
from pool.storage_mode import PoolStorageMode
from pool.task_queues import (
    ACTIVITY_WORKER_POD_NAME_ENV,
    POOL_WORKER_NAME,
    activity_worker_name_suffix,
    daemon_dispatch_queue,
    is_activity_scoped_pod,
)
from rest.backends.data_backend.connector_activity_registry import (
    validate_connector_activity_registration,
)
from service_framework.deploy_env_config import deploy_env
from service_framework.env_var import EnvVar
from temporalio_commons.runtime_utils import init_temporal_runtime
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_registry import WorkerEnv, WorkerOptions

log = logging.getLogger(__name__)

_SIGNAL_SHUTDOWN_TIMEOUT_SECS = 25.0
_ACTIVITY_SHUTDOWN_GRACE_SECS = 5.0

_POOL_WORKER_UID = EnvVar(
    'POOL_WORKER_UID',
    str,
    required=True,
    description='Stable identity reported in heartbeats.',
)
_POOL_STORAGE_MODE = EnvVar(
    'POOL_STORAGE_MODE',
    PoolStorageMode,
    default_value=PoolStorageMode.DAEMON_ON_PVC,
    description='See pool.storage_mode.PoolStorageMode.',
)
_POOL_TASK_QUEUE = EnvVar(
    'POOL_TASK_QUEUE',
    str,
    required=True,
    description='Temporal task queue this worker polls.',
)
_POOL_GPU_CONCURRENCY = EnvVar(
    'POOL_GPU_CONCURRENCY',
    int,
    default_value=1,
    description='GPU semaphore size; gates concurrent train/predict.',
)
_POOL_MAX_CONCURRENT_ACTIVITIES = EnvVar(
    'POOL_MAX_CONCURRENT_ACTIVITIES',
    int,
    default_value=0,
    description=('Worker.max_concurrent_activities cap; 0 = unlimited. '
                 'Forced to 1 for activity-scoped workers.'),
)
_POOL_NOFILE_SOFT_LIMIT = EnvVar(
    'POOL_NOFILE_SOFT_LIMIT',
    int,
    default_value=65535,
    description='Minimum soft open-file limit for pool worker processes.',
)

_POOL_ACTIVITY_WORKER_POD_NAME = EnvVar(
    ACTIVITY_WORKER_POD_NAME_ENV,
    str,
    default_value='',
    description=(
        'Kubernetes pod name injected for activity-scoped worker identity.'),
)

# Dump stacktrace on segfault (useful for native-extension crashes).
faulthandler.enable()


def _configure_nofile_soft_limit() -> int:
    """Raise the worker soft nofile limit before spawning child processes."""
    requested = _POOL_NOFILE_SOFT_LIMIT.parse_value()
    if requested <= 0:
        raise ValueError('POOL_NOFILE_SOFT_LIMIT must be greater than zero')

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= requested:
        log.info(
            'pool worker nofile limit unchanged: soft=%s hard=%s requested=%d',
            soft,
            hard,
            requested,
        )
        return soft

    if hard != resource.RLIM_INFINITY and hard < requested:
        raise RuntimeError(
            f'POOL_NOFILE_SOFT_LIMIT={requested} was requested, but the hard '
            f'open-file limit is {hard}; configure the container runtime with '
            f'a hard nofile limit of at least {requested}.')

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (requested, hard))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f'Failed to raise the pool worker soft open-file limit from '
            f'{soft} to {requested} (hard={hard}).') from exc

    effective, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if effective < requested:
        raise RuntimeError(
            f'Pool worker soft open-file limit remained {effective} after '
            f'requesting {requested}.')
    log.info(
        'raised pool worker soft open-file limit: %d -> %d (hard=%s)',
        soft,
        effective,
        hard,
    )
    return effective


def _discover_activities(
        snapshot_activities: SnapshotActivities) -> List[Callable]:
    """Collect ``@activity.defn``-decorated methods on the instance."""
    found: List[Callable] = []
    for name, member in inspect.getmembers(snapshot_activities,
                                           predicate=callable):
        if name.startswith('_'):
            continue
        try:
            activity._Definition.must_from_callable(member)
        except TypeError:
            # ``must_from_callable`` raises TypeError for callables that
            # aren't @activity.defn-decorated. Narrower than ``Exception``
            # so genuine registration bugs (import errors, attribute
            # lookup failures) surface here instead of silently producing
            # a worker with fewer activities than expected.
            continue
        found.append(member)
    validate_connector_activity_registration(found)
    return found


def _graceful_shutdown_timeout(storage_mode: PoolStorageMode) -> timedelta:
    # Activity-scoped workers initiate shutdown after their one activity body
    # returns. Preserve a bounded result-report flush window there while
    # leaving 25s of the pod's 30s termination grace for cancellation and the
    # native supervisor. Long-lived daemons stop only on process signals, so
    # delaying cancellation would consume the container termination grace.
    if storage_mode.is_activity_scoped:
        return timedelta(seconds=_ACTIVITY_SHUTDOWN_GRACE_SECS)
    return timedelta()


async def _run_worker_until_shutdown(
    worker: Worker,
    *,
    activity_done: Optional[asyncio.Event] = None,
    signal_requested: Optional[asyncio.Event] = None,
) -> bool:
    """Run a Temporal worker until it exits or an orderly stop is requested.

    Python runs as PID 1 in the pool-worker containers, where an unhandled
    SIGTERM is ignored. Convert TERM/INT into ``Worker.shutdown()`` so Temporal
    cancels in-flight activities and their owned subprocess sessions before
    the container exits.
    """
    loop = asyncio.get_running_loop()
    signal_request: asyncio.Future[int] = loop.create_future()
    if signal_requested is None:
        signal_requested = asyncio.Event()
    installed_signals: list[int] = []

    def _request_signal_shutdown(signum: int) -> None:
        signal_requested.set()
        if not signal_request.done():
            signal_request.set_result(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, _request_signal_shutdown, signum)
            installed_signals.append(signum)
    except Exception:
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        raise

    worker_task = asyncio.create_task(worker.run(), name='pool-worker-run')
    activity_done_task: 'asyncio.Task[bool] | None' = None
    if activity_done is not None:
        activity_done_task = asyncio.create_task(
            activity_done.wait(), name='pool-self-exit-watcher')

    waiters: list[asyncio.Future[Any]] = [worker_task, signal_request]
    if activity_done_task is not None:
        waiters.append(activity_done_task)

    try:
        done, _ = await asyncio.wait(waiters,
                                     return_when=asyncio.FIRST_COMPLETED)
        if worker_task in done:
            await worker_task
            return signal_request.done()

        if signal_request in done:
            signum = signal_request.result()
            log.info(
                'pool worker_main: received %s; initiating Temporal worker '
                'shutdown.', signal.Signals(signum).name)
        else:
            log.info('pool worker_main: activity_done fired; '
                     'initiating Temporal worker shutdown.')

        if signal_request in done:
            # The daemon Worker is configured with zero Core grace, so this
            # immediately requests cancellation of active activities. Keep an
            # application deadline below the 30s container/pod grace; native
            # process cleanup itself is bounded to 10s TERM + 5s KILL/reap.
            await asyncio.wait_for(
                worker.shutdown(), timeout=_SIGNAL_SHUTDOWN_TIMEOUT_SECS)
        else:
            await worker.shutdown()
        await worker_task
        return signal_request.done()
    finally:
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        if not signal_request.done():
            signal_request.cancel()
        if activity_done_task is not None and not activity_done_task.done():
            activity_done_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await activity_done_task
        if not worker_task.done():
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task


def _exit_after_signal_shutdown(exit_code: int = 0) -> None:
    """Exit without joining native work abandoned by activity cancellation."""
    log.info('pool worker_main: Temporal shutdown and activity cleanup '
             'complete; exiting signal-stopped worker with code %d.',
             exit_code)
    logging.drain_logs()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError):
            stream.flush()
    os._exit(exit_code)


async def _finalize_worker(
    worker_env: WorkerEnv,
    activities: SnapshotActivities,
    *,
    signal_shutdown: bool,
    signal_shutdown_failed: bool = False,
) -> None:
    """Release worker resources before the signal-only process exit."""
    cleanup_errors: List[BaseException] = []
    try:
        worker_env.shutdown()
    except BaseException as error:
        log.exception('pool worker_main: WorkerEnv shutdown failed.')
        cleanup_errors.append(error)
    try:
        await activities.stop()
    except BaseException as error:
        log.exception('pool worker_main: activity cleanup failed.')
        cleanup_errors.append(error)

    if signal_shutdown:
        _exit_after_signal_shutdown(
            1 if signal_shutdown_failed or cleanup_errors else 0)
        return
    if cleanup_errors:
        raise cleanup_errors[0]


async def main() -> None:
    init_logging_and_tracing(POOL_WORKER_NAME)
    _configure_nofile_soft_limit()

    deploy_config = deploy_env()

    pool_worker_uid = _POOL_WORKER_UID.parse_value()
    task_queue = _POOL_TASK_QUEUE.parse_value()
    storage_mode = _POOL_STORAGE_MODE.parse_value()
    gpu_concurrency = _POOL_GPU_CONCURRENCY.parse_value()
    activity_pod_name = _POOL_ACTIVITY_WORKER_POD_NAME.parse_value()

    # Activity-scoped workers are hard-capped at 1: prevents Temporal from
    # pulling a second task during the shutdown-drain window.
    if storage_mode.is_activity_scoped:
        max_concurrent_activities: Optional[int] = 1
    else:
        configured = _POOL_MAX_CONCURRENT_ACTIVITIES.parse_value()
        max_concurrent_activities = configured if configured > 0 else None

    # The data persist dir is derived from deploy_env() on demand
    # (kumo.data.get_persist_dir); no process-global state to prime here.
    from kumo.data import get_persist_dir
    log.info('pool worker_main: data backend derived at %s',
             get_persist_dir())

    controller = get_controller_client(deploy_config)
    # Activity-scoped pods sharing a daemon's PVC delegate ensure_snapshot
    # writes back to that daemon — see Snapshot Mutation Invariant.
    temporal_client: Optional[TemporalClient] = (
        TemporalClient(deploy_config) if is_activity_scoped_pod(
            pool_worker_uid, storage_mode) else None)
    if temporal_client is not None:
        log.info(
            'pool worker_main: activity-scoped pod (uid=%s, storage_mode=%s); '
            'ensure_snapshot delegates to daemon at task_queue=%s',
            pool_worker_uid, storage_mode.value,
            daemon_dispatch_queue(pool_worker_uid))
    activities = SnapshotActivities(
        pool_worker_uid=pool_worker_uid,
        controller=controller,
        storage_mode=storage_mode,
        gpu_concurrency=gpu_concurrency,
        temporal_client=temporal_client,
    )
    await activities.start()

    discovered = _discover_activities(activities)

    from rest.backends.data_backend.activities import (
        diskgraph_ingest_stats_activity,
    )
    from rest.backends.data_backend.activities.diskgraph_ingest_activity import (
        cleanup_source_files,
    )
    discovered = discovered + [
        diskgraph_ingest_stats_activity,
        cleanup_source_files,
    ]

    log.info(
        'pool worker_main starting (pid=%d): uid=%s tq=%s storage_mode=%s '
        'activities=%s gpu=%d', os.getpid(), pool_worker_uid, task_queue,
        storage_mode.value, [a.__name__ for a in discovered], gpu_concurrency)

    worker_deploy_config = dataclasses.replace(deploy_config,
                                               client_id=POOL_WORKER_NAME)
    # max_worker_threads gives build_worker a ThreadPoolExecutor for sync
    # activities (e.g., download_table_files, diskgraph_ingest_stats_activity).
    worker_env = WorkerEnv(
        worker_name=POOL_WORKER_NAME,
        options=WorkerOptions(max_worker_threads=8),
        deploy_config=worker_deploy_config,
    )
    # Activity-scoped shutdown needs 30s for the completed activity result to
    # reach the cluster. Long-lived daemons use 0s so TERM immediately cancels
    # active work and leaves the deployment grace for native process cleanup.
    worker = await worker_env.build_worker(
        activities=discovered,
        workflows=[],
        alt_task_queue=task_queue,
        worker_name_suffix=activity_worker_name_suffix(activity_pod_name),
        max_concurrent_activities=max_concurrent_activities,
        graceful_shutdown_timeout=_graceful_shutdown_timeout(storage_mode),
    )

    # Now that worker init is done (Temporal SDK pre-fetches build_id via
    # synchronous file I/O), start the event-loop stall watchdog. Startup is
    # once-only so steady-state stalls past this point are what we want to
    # catch.
    install_loop_guards()

    signal_requested = asyncio.Event()
    signal_shutdown = False
    signal_shutdown_failed = False
    try:
        signal_shutdown = await _run_worker_until_shutdown(
            worker,
            activity_done=(activities.activity_done
                           if storage_mode.is_activity_scoped else None),
            signal_requested=signal_requested,
        )
    except BaseException:
        signal_shutdown_failed = signal_requested.is_set()
        raise
    finally:
        # ``asyncio.run`` waits up to five minutes for default-executor
        # threads. Native DiskGraph calls cannot be stopped by cancelling
        # their ``to_thread`` awaiter, so a signal-stopped worker exits only
        # after Temporal and activity cleanup are done.
        await _finalize_worker(worker_env, activities,
                               signal_shutdown=(signal_shutdown
                                                or signal_requested.is_set()),
                               signal_shutdown_failed=signal_shutdown_failed)

    # Propagate activity failure as a non-zero exit so the K8s Job's
    # OnFailure restart triggers and Temporal retries onto the new pod.
    # TODO: alt design — keep the pod polling for the retry attempt
    # instead of exiting. Saves the restart overhead but gives up the
    # crash-isolation rationale of ACTIVITY_SCOPED. Plausibly fine for
    # CPU activities (gen_tt / materialize_*); risky for GPU
    # train / predict where leaked CUDA state often correlates with the
    # failure. Revisit once we have failure-mode data from real runs.
    if storage_mode.is_activity_scoped and activities.activity_failed:
        sys.exit(1)


if __name__ == '__main__':
    init_temporal_runtime(POOL_WORKER_NAME)
    asyncio.run(main())
