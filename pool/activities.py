# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool worker activities.

:class:`SnapshotActivities` is instantiated once per worker process;
the worker bootstrap (see ``pool.worker_main``) registers whichever
subset of its ``@activity.defn`` methods that process serves.
"""
import asyncio
import contextlib
import contextvars
import copy
import dataclasses
import json
import logging as _logging
import os
import os.path as osp
import shutil
import signal
import sys
import tempfile
import time
import uuid
from asyncio import AbstractEventLoop
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Tuple

import tenacity
from kumo_core.diskgraph.snapshot import DiskgraphIngestResult
from kumo_core.train.artifact_stats import (
    PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES,
    PREDICTION_ARTIFACT_STATS_PATH_ENV,
    PREDICTION_ARTIFACT_STATS_REQUEST_ENV,
    PredictionArtifactStatsPayload,
    PredictionArtifactStatsRequest,
)
from temporalio import activity
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.exceptions import ApplicationError

from common_utils.error import PublicException
from io_utils import fsutil, logging, parquet_util
from kumo.connector.artifact_staging import local_parquet_files
from kumo.data import get_persist_dir
from kumo.data.utils import (
    get_diskgraph_snapshot_ingest_dir,
    get_diskgraph_snapshot_ingest_path,
)
from kumo.diskgraph import DiskgraphHandle, get_service
from kumo.diskgraph.errors import DiskgraphNoSpaceError
from kumo.diskgraph.job_sidecars import (
    prediction_materialization_ownership,
    restore_job_sidecars_for_use_from_config_path,
    upload_job_sidecars_from_config_path,
)
from kumo.train.prediction_partitions import (
    prediction_output_partition_range,
    split_partitions_across_jobs,
)
from pool.client import PoolControllerClient, get_controller_client
from pool.dataclasses import (
    JobID,
    PlacementRequest,
    PlacementResult,
    PoolStats,
    PoolStatsFilter,
    SnapshotID,
    WorkerEvent,
    WorkerEventType,
    WorkerStateSnapshot,
    WorkerStatusReport,
)
from pool.diskgraph_args import DiskgraphIngestArgs
from pool.locks import PerSnapshotLocks
from pool.pidfd import pidfd_open, pidfd_process_exited, pidfd_send_signal
# Single source of truth for the heartbeat cadence — controller side owns it.
from pool.service import POOL_HEARTBEAT_INTERVAL_SECS
from pool.snapshot_paths import normalized_diskgraph_snapshot_id_from_path
from pool.storage_mode import PoolStorageMode
from pool.task_queues import daemon_dispatch_queue
from pool.training_progress import (
    TRAINING_PROGRESS_PUBLISH_TIMEOUT_SECS,
    publish_training_progress,
)
from rest.backends.data_backend import connector_activity_registry
from rest.backends.ml_backend.activities.common import update_job_status
from rest.backends.ml_backend.activities.generate_target_table import (
    generate_prediction_table,
    generate_training_table,
)
from rest.backends.ml_backend.activities.materialize_prediction_task import (
    build_ttng_custom_pred_table_bundle,
    materialize_prediction_task,
)
from rest.backends.ml_backend.activities.materialize_training_task import (
    build_ttng_custom_train_table_bundle,
    materialize_training_task,
)
from rest.backends.ml_backend.ml_job_output_store import MLJobOutputStore
from rest.backends.ml_backend.timing import kumo_timing
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    DatabricksVolumeShard,
    PredictionShardResult,
    requested_prediction_artifact_types,
)
from rest.config.ml_backend import JobState
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import DataSnapshot, JobEventLogEntry, JobType
from statedb.utils.errors import PredictionOutputProcessingError
from temporalio_commons.auto_rpc_exception import RpcError as _RpcError
from temporalio_commons.process_pool_utils import run_with_async_caller_loop
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_registry import RunSingleActivityWorkflow

log = logging.getLogger(__name__)

_DATABRICKS_PARQUET_FILE_TARGET_BYTES = 128 << 20

PERMANENT_SNAPSHOT_MISS_ERROR_TYPE = 'PermanentSnapshotMissError'
IN_PROGRESS_MARKER = '.in_progress'
SUCCESS_MARKER = '_SUCCESS'
RESERVED_DAEMON_DIR = '.pool_worker_daemon'

# Upper bound on a daemon-side ``ensure_snapshot`` invocation: SSD-residency
# check, dataplane restore, and worst-case fresh ingest. Matches
# ``DiskgraphIngestWorkflow``'s schedule_to_close_timeout.
_DAEMON_ENSURE_SNAPSHOT_TIMEOUT = timedelta(hours=4)

_DISK_PROBE_WAIT_SECS = 1.0
_DISK_TELEMETRY_MAX_AGE_SECS = 2 * POOL_HEARTBEAT_INTERVAL_SECS

_DiskProbeResult = Tuple[int, int, Optional[int]]
_DiskTelemetry = Tuple[_DiskProbeResult, _DiskProbeResult]
_UNKNOWN_DISK_PROBE: _DiskProbeResult = (0, 0, None)
_UNKNOWN_DISK_TELEMETRY: _DiskTelemetry = (_UNKNOWN_DISK_PROBE,
                                           _UNKNOWN_DISK_PROBE)
_BACKGROUND_DISKGRAPH_CLEANUPS: set[asyncio.Task] = set()


def _load_prediction_artifact_stats_payload(
    path: str,
    request: PredictionArtifactStatsRequest,
) -> PredictionArtifactStatsPayload:
    """Read and validate the producer's attempt-local bounded payload."""
    size = fsutil.filesize(path)
    if (size is None or size <= 0
            or size > PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES):
        raise ValueError(
            'Prediction artifact stats payload has an invalid size')
    with fsutil.get_binary_handle(path, 'rb') as handle:
        encoded = handle.read(PREDICTION_ARTIFACT_STATS_MAX_PAYLOAD_BYTES + 1)
        if len(encoded) != size or handle.read(1):
            raise ValueError(
                'Prediction artifact stats payload changed while reading')
    payload = PredictionArtifactStatsPayload.from_json(encoded.decode())
    payload.validate(request)
    return payload


def _schedule_activity_heartbeat(activity_loop: AbstractEventLoop) -> None:
    activity_loop.call_soon_threadsafe(activity.heartbeat)


def _probe_local_disk(path: Optional[str]) -> _DiskProbeResult:
    """Return total, used, and device for a local path, or unknown values."""
    if path is None:
        return 0, 0, None
    try:
        if not fsutil.is_local(path):
            return 0, 0, None
        current = Path(path)
        while True:
            try:
                device = current.stat().st_dev
                usage = shutil.disk_usage(current)
                return usage.total, usage.total - usage.free, device
            except FileNotFoundError:
                parent = current.parent
                if parent == current:
                    return 0, 0, None
                current = parent
    except (OSError, ValueError):
        log.debug('Local disk probe failed for %s; reporting unknown', path,
                  exc_info=True)
        return 0, 0, None


def _probe_local_disks(
    data_root: Optional[str],
    dataplane_root: Optional[str],
) -> _DiskTelemetry:
    """Probe both report roots as one non-raising telemetry sample."""
    try:
        return (_probe_local_disk(data_root), _probe_local_disk(dataplane_root))
    except Exception:
        log.debug('Combined local disk probe failed; reporting unknown',
                  exc_info=True)
        return _UNKNOWN_DISK_TELEMETRY


def _retrieve_task_exception(task: asyncio.Future) -> None:
    """Mark a late task exception retrieved without changing owner state."""
    if not task.cancelled():
        task.exception()


def _cleanup_diskgraph_paths(cleanup_dir: Optional[str]) -> None:
    """Best-effort cleanup for paths owned by an off-loop DiskGraph call."""
    if cleanup_dir is not None:
        try:
            _rmtree_if_present(cleanup_dir)
        except Exception:
            log.warning('Failed to clean up DiskGraph directory %s',
                        cleanup_dir, exc_info=True)


def _schedule_diskgraph_cleanup(
    task: asyncio.Future,
    *,
    cleanup_dir: Optional[str],
) -> None:
    """Clean abandoned outputs off-loop after their owner task finishes."""
    if cleanup_dir is None:
        return

    def schedule_cleanup(_task: asyncio.Future) -> None:
        cleanup_task = asyncio.create_task(
            asyncio.to_thread(_cleanup_diskgraph_paths, cleanup_dir),
            name='diskgraph-abandonment-cleanup',
        )
        _BACKGROUND_DISKGRAPH_CLEANUPS.add(cleanup_task)

        def retire(completed: asyncio.Task) -> None:
            _BACKGROUND_DISKGRAPH_CLEANUPS.discard(completed)
            _retrieve_task_exception(completed)

        cleanup_task.add_done_callback(retire)

    task.add_done_callback(schedule_cleanup)


def _rmtree_if_present(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _rmtree_if_present_strict(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _upload_prediction_artifact_to_databricks(
    connector_spec: Any,
    *,
    artifact_path: str,
    artifact_type: Any,
    job_id: JobID,
    output_index: int,
    shard_index: int,
    attempt_id: str,
    num_shards: int = 1,
) -> DatabricksVolumeShard:
    """Upload one producer shard directly from its GPU worker."""
    from kumo.connector import DatabricksConnector
    from kumo.train.artifact_manifest import prediction_shard_producer_abi
    from rest.utils.driver_common import get_connector_from_spec

    connector = get_connector_from_spec(connector_spec)
    if not isinstance(connector, DatabricksConnector):
        raise TypeError('Databricks prediction output has a non-Databricks '
                        'connector')
    with connector:
        connector.ensure_managed_volume()
        metadata = parquet_util.get_parquet_metadata(artifact_path)
        if metadata.num_rows <= 0:
            raise PredictionOutputProcessingError(
                message=('Batch prediction produced an empty '
                         f'{artifact_type.value.lower()} artifact.'),
                non_retryable=True,
            )
        local_files = local_parquet_files(artifact_path)
        uploaded_bytes = sum(path.stat().st_size for path in local_files)
        producer_abi = prediction_shard_producer_abi(num_shards)
        stage_key = (f'batch-prediction:{job_id}:{output_index}:'
                     f'{artifact_type.value}')
        if num_shards > 1:
            # Older finalizers reject mixed stage paths.
            stage_key += f':{producer_abi}'
        stage = connector.managed_volume_stage(stage_key)
        file_prefix = f'shard-{shard_index:05d}-attempt-{attempt_id}'
        upload_started = time.monotonic()
        _, files = connector.upload_local_artifacts_to_volume(
            artifact_path,
            stage=stage,
            file_prefix=file_prefix,
            files=local_files,
        )
        upload_seconds = max(time.monotonic() - upload_started, 0.001)
        log.info(
            'Databricks managed-volume upload complete: job=%s '
            'output_index=%s shard=%s files=%s bytes=%s seconds=%.3f '
            'throughput_mib_s=%.1f',
            job_id,
            output_index,
            shard_index,
            len(files),
            uploaded_bytes,
            upload_seconds,
            uploaded_bytes / upload_seconds / (1 << 20),
        )
        return DatabricksVolumeShard(
            output_index=output_index,
            artifact_type=artifact_type,
            stage_path=stage,
            staged_files=files,
            num_rows=metadata.num_rows,
            num_cols=metadata.num_cols,
            size_bytes=int(round(metadata.avg_row_size * metadata.num_rows)),
            uploaded_bytes=uploaded_bytes,
            upload_seconds=upload_seconds,
            file_count=len(files),
            producer_abi=producer_abi,
        )


def _persist_prediction_artifacts(
    local_output_dir: str,
    durable_output_dir: str,
) -> None:
    """Merge one worker's uniquely named Parquet parts into Kumo storage."""
    for artifact_name in os.listdir(local_output_dir):
        source = osp.join(local_output_dir, artifact_name)
        destination = osp.join(durable_output_dir, artifact_name)
        fsutil.copy(
            source,
            destination,
            overwrite=True,
            raise_on_exists=False,
        )


def _cleanup_databricks_prediction_upload(
    connector_spec: Any,
    files: list[str],
) -> None:
    from kumo.connector import DatabricksConnector
    from rest.utils.driver_common import get_connector_from_spec

    connector = get_connector_from_spec(connector_spec)
    if not isinstance(connector, DatabricksConnector):
        return
    try:
        connector.cleanup_managed_volume_files(files)
    finally:
        connector.close()


async def _run_diskgraph_async(
    fn,
    *args,
    abandonment_cleanup_dir=None,
    join_on_cancel=False,
    cancellation_cleanup_dir=None,
    **kwargs,
):
    """Run synchronous DiskGraph work from an async activity.

    The caller loop is stamped on the worker thread so the standalone ingest
    supervisor and remaining process-isolated native calls can route
    heartbeats with ``call_soon_threadsafe``.
    """
    loop = asyncio.get_running_loop()

    def submit_off_loop(callback):
        # Match ``asyncio.to_thread`` context propagation without creating a
        # Task that runner-wide shutdown can terminally cancel while its
        # executor thread is still writing.
        context = contextvars.copy_context()
        return loop.run_in_executor(None, context.run, callback)

    def run() -> Any:
        return run_with_async_caller_loop(loop, fn, *args, **kwargs)

    task = submit_off_loop(run)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        if join_on_cancel:
            # An executor Future cannot stop a synchronous connector call.
            # Keep the caller's per-snapshot lock held until that call and its
            # cleanup are both terminal; otherwise a retry can race the old
            # writer or have its new source tree deleted by a late callback.
            current_task = asyncio.current_task()
            _uncancel_task(current_task)
            started = time.monotonic()

            async def _join(task_to_join, description):
                while True:
                    try:
                        await asyncio.shield(task_to_join)
                    except asyncio.CancelledError:
                        _uncancel_task(current_task)
                        log.info(
                            'Repeated cancellation while joining abandoned '
                            'DiskGraph %s; continuing cleanup.', description)
                        continue
                    except BaseException as exc:
                        log.warning(
                            'DiskGraph %s failed after its caller was '
                            'cancelled; preserving cancellation as the public '
                            'outcome.',
                            description,
                            exc_info=(type(exc), exc, exc.__traceback__))
                    break

            await _join(task, 'operation')
            if cancellation_cleanup_dir is not None:
                def cleanup_cancelled_source() -> None:
                    _rmtree_if_present_strict(cancellation_cleanup_dir)

                cleanup_task = submit_off_loop(cleanup_cancelled_source)
                await _join(cleanup_task, 'source cleanup')
            log.info(
                'Cancelled DiskGraph operation joined and cleaned after '
                '%.3fs.', time.monotonic() - started)
            raise cancelled
        _schedule_diskgraph_cleanup(task,
                                    cleanup_dir=abandonment_cleanup_dir)
        raise


def _uncancel_task(task) -> None:
    """Clear repeated cancellation on Python versions that support it."""
    uncancel = getattr(task, 'uncancel', None)
    if uncancel is not None:
        uncancel()


def _materialize_and_upload_prediction_task(
    snapshot_id: str,
    job_input: Any,
    settings: Any,
    serving: bool,
    pred_config_path: str,
    home_dir: Optional[str],
    data_persist_dir: str,
) -> Tuple[str, str, Any]:
    """Materialize and publish one BP task under cross-process ownership."""
    from kumo_core.job_config import PredictionConfig

    config = PredictionConfig.load(pred_config_path)
    index_path = config.diskgraph_index_path
    if not index_path:
        # Preserve the materializer's precise config error. If an unexpected
        # implementation returns successfully, fail closed rather than report
        # success without publishing its sidecars.
        materialize_prediction_task(job_input, settings, serving,
                                    pred_config_path, home_dir)
        raise ValueError(
            'Prediction materialization completed without '
            'diskgraph_index_path; sidecars cannot be safely published')
    with prediction_materialization_ownership(index_path, str(job_input.id)):
        result = materialize_prediction_task(job_input, settings, serving,
                                             pred_config_path, home_dir)
        upload_job_sidecars_from_config_path(
            snapshot_id,
            result[0],
            prediction=True,
            data_persist_dir=data_persist_dir,
        )
        return result


def _build_diskgraph_ingest_config(ingest_args: DiskgraphIngestArgs):
    """Hydrate connector configuration only in the pool activity process."""
    from rest.config.graph_v2 import GraphDefinition
    from rest.driver.data_snapshot_api_driver import _build_ingest_graph_config
    from rest.driver.graph_logic import _build_graph_from_definition_with_specs
    from statedb.config.common import TableName

    graph_definition = GraphDefinition(
        tables=ingest_args.all_graph_tables,
        col_groups=ingest_args.graph_definition.col_groups,
    )
    graph = _build_graph_from_definition_with_specs(
        graph_definition,
        ingest_args.graph_display_name,
        ingest_args.connector_specs.values(),
    )
    table_versions = {
        TableName(name): str(version)
        for name, version in ingest_args.table_data_versions.items()
    }
    graph_config = _build_ingest_graph_config(graph, table_versions,
                                              graph_definition)
    # Source files were already downloaded above and ``service.ingest`` always
    # receives their explicit paths. Connector configs are therefore neither
    # needed by DiskGraph nor safe to cross its spawned-process boundary:
    # torch-geometric registers connector config classes dynamically, which
    # standard multiprocessing pickle cannot import in the child.
    graph_config.data_source_dict = {}
    return graph_config


async def _snapshot_dir_size_bytes(snapshot_id: SnapshotID,
                                   local_dir: str) -> int:
    size_bytes = await asyncio.to_thread(fsutil.dirsize, local_dir)
    if size_bytes is None or size_bytes <= 0:
        raise RuntimeError(
            f'Could not compute positive local snapshot size for '
            f'snapshot={snapshot_id} at {local_dir}: {size_bytes}')
    return int(size_bytes)


def _iter_local_snapshot_dirs(
        local_ssd_root: str) -> list[tuple[SnapshotID, Path]]:
    root = Path(local_ssd_root)
    root.mkdir(parents=True, exist_ok=True)
    snapshots: list[tuple[SnapshotID, Path]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if (not child.is_dir() or child.name == RESERVED_DAEMON_DIR
                or child.name.startswith('.')):
            continue
        if '@' not in child.name:
            try:
                version_dirs = sorted(child.iterdir(), key=lambda p: p.name)
            except (PermissionError, NotADirectoryError) as exc:
                # Not every top-level entry under the data root is a snapshot
                # dir we can scan. ext4 volumes (e.g. Azure managed-disk PVCs)
                # create a root-owned, 0700 ``lost+found`` at the mount root
                # that the non-root worker cannot ``scandir``; a stray file
                # can likewise fail ``iterdir``. Skip such entries instead of
                # aborting daemon startup on the first unreadable one.
                log.warning(
                    'Skipping unreadable entry under snapshot root: %s (%s)',
                    child, exc)
                continue
            nested_snapshots: list[tuple[SnapshotID, Path]] = []
            for version_dir in version_dirs:
                if not version_dir.is_dir() or version_dir.name.startswith('.'):
                    continue
                snapshot_id = normalized_diskgraph_snapshot_id_from_path(
                    child.name, version_dir.name)
                if snapshot_id is not None:
                    nested_snapshots.append((snapshot_id, version_dir))
            if nested_snapshots:
                snapshots.extend(nested_snapshots)
                continue
            continue
        graph_key, _, graph_data_version = child.name.partition('@')
        snapshot_id = normalized_diskgraph_snapshot_id_from_path(
            graph_key, graph_data_version)
        if snapshot_id is not None:
            snapshots.append((snapshot_id, child))
    return snapshots


def _is_snapshot_ready(snapshot_dir: Path) -> bool:
    return (snapshot_dir / SUCCESS_MARKER).exists()


def _remove_empty_snapshot_parent(snapshot_dir: Path, root: Path) -> None:
    parent = snapshot_dir.parent
    if parent == root:
        return
    try:
        parent.rmdir()
    except OSError:
        return


def _clean_in_progress_snapshots(local_ssd_root: str) -> list[SnapshotID]:
    root = Path(local_ssd_root)
    root.mkdir(parents=True, exist_ok=True)
    cleaned: list[SnapshotID] = []
    for snapshot_id, snapshot_path in _iter_local_snapshot_dirs(str(root)):
        if _is_snapshot_ready(snapshot_path):
            continue
        shutil.rmtree(snapshot_path, ignore_errors=True)
        _remove_empty_snapshot_parent(snapshot_path, root)
        cleaned.append(snapshot_id)
    return cleaned


def _visible_gpus() -> list[str]:
    import torch

    # Torch's count stops at an invalid entry, as CUDA does.
    count = torch.cuda.device_count()
    listed = os.environ.get('CUDA_VISIBLE_DEVICES')
    return ([device.strip() for device in listed.split(',')]
            if listed is not None else [str(i) for i in range(count)])[:count]


class _GpuSlots:
    """Bounds concurrent GPU activities, pinning each to its own GPU."""

    def __init__(self, count: int,
                 devices: Optional[list[str]] = None) -> None:
        if count < 1:
            raise ValueError(f'GPU concurrency must be >= 1, got {count}')
        self._free: asyncio.Queue[int] = asyncio.Queue()
        for slot in range(count):
            self._free.put_nowait(slot)
        self._devices: list[str] = []
        if count > 1:
            self._devices = _visible_gpus() if devices is None else devices
        if 0 < len(self._devices) < count:
            raise ValueError(
                f'GPU concurrency {count} needs one GPU per slot, but this '
                f'worker sees {len(self._devices)} GPU(s)')

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[dict[str, str]]:
        slot = await self._free.get()
        try:
            if not self._devices:
                yield {}
            else:
                log.info('GPU slot %d -> CUDA_VISIBLE_DEVICES=%s', slot,
                         self._devices[slot])
                yield {'CUDA_VISIBLE_DEVICES': self._devices[slot]}
        finally:
            self._free.put_nowait(slot)


class SnapshotActivities:
    """Single-instance host for every diskgraph-pool Temporal activity.

    See module docstring for the role / registration story. Construct
    once at worker startup, call :meth:`start` to bring it online (initial
    heartbeat + optional periodic loop), register the role-appropriate
    bound methods with a Temporal :class:`Worker`, run the worker, and
    call :meth:`stop` on shutdown.
    """
    def __init__(
        self,
        *,
        pool_worker_uid: str,
        controller: PoolControllerClient,
        storage_mode: PoolStorageMode = PoolStorageMode.DAEMON_ON_PVC,
        gpu_concurrency: int = 1,
        ingest_concurrency: int = 2,
        temporal_client: Optional[TemporalClient] = None,
    ) -> None:
        self._uid = pool_worker_uid
        self._controller = controller
        self._storage_mode = storage_mode
        # Self-contained store for the job-output StateDB writes this worker
        # makes, both on the worker's event loop: publishing kumo-auto training
        # progress (``train``) and the "materializing" status the
        # ``materialize_prediction_task`` wrapper writes before dispatching the
        # diskgraph work. Created on first use (see ``_get_job_output_store``);
        # it opens one StateDB on the first write and reuses it, and ``stop``
        # closes it. Construction itself does no I/O, so a pure-ingest daemon
        # pod that runs neither never touches StateDB.
        self._job_output_store: Optional[MLJobOutputStore] = None
        self._snapshot_locks = PerSnapshotLocks()
        self._gpu_slots = _GpuSlots(gpu_concurrency)
        # Caps concurrent fresh Rust ingests across DIFFERENT snapshots on
        # this worker — CPU + NVMe contention bound. Same-snapshot
        # serialization is the per-snapshot lock's job.
        self._ingest_semaphore = asyncio.Semaphore(ingest_concurrency)
        self._activity_done = asyncio.Event()
        self._activity_failed = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._status_report_lock = asyncio.Lock()
        self._disk_probe_task: Optional[asyncio.Task[_DiskTelemetry]] = None
        self._disk_telemetry = _UNKNOWN_DISK_TELEMETRY
        self._disk_telemetry_at: Optional[float] = None
        self._disk_probe_started_at: Optional[float] = None
        self._disk_probe_timeout_warned = False
        self._monotonic = time.monotonic
        # Set only on activity-scoped pods sharing a daemon's PVC; used
        # by ``_ensure_snapshot`` to delegate writes to that daemon.
        self._temporal_client = temporal_client
        # In-memory ledger of snapshot IDs currently present on this
        # worker's local SSD with a valid ``_SUCCESS`` marker. Seeded
        # by ``start()`` from disk, then maintained inline by
        # ``_ensure_snapshot_local`` and ``delete_snapshot`` so we
        # don't rescan the data root on every heartbeat tick. Only
        # mutated under the per-snapshot lock, so set ops are safe.
        self._ready_snapshots: set = set()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Bring the worker online: emit one initial heartbeat (so the
        controller registers this worker as alive), then start the
        periodic heartbeat task on long-lived workers.

        Idempotent against re-entry while running: the second call is a
        no-op (the heartbeat task is already created). Idempotent against
        re-entry after :meth:`stop`: a fresh heartbeat task is created.
        """
        # Seed ``_ready_snapshots`` once from disk so the very first
        # heartbeat reports an accurate WARM set (worker may have been
        # restarted with a populated PVC). After this, transitions in
        # ``_ensure_snapshot_local`` keep it in sync without rescans.
        # Daemons only — activity-scoped pods either share the
        # daemon's PVC (where the daemon is the source of truth) or
        # write to ``emptyDir`` that doesn't survive the pod, so
        # advertising their snapshots as WARM would mislead the
        # controller.
        if self._storage_mode.is_long_lived:
            from service_framework._raw_deploy_config import DiskgraphConfig
            data_compute = deploy_env().data_compute
            if isinstance(data_compute, DiskgraphConfig):
                cleaned_snapshot_ids = await asyncio.to_thread(
                    _clean_in_progress_snapshots, data_compute.data_root)
                if cleaned_snapshot_ids:
                    log.info(
                        'SnapshotActivities.start: cleaned in-progress '
                        'snapshots at startup: %s', cleaned_snapshot_ids)
            handles = await get_service().list_ready_handles.coroutine()
            self._ready_snapshots = {
                DataSnapshot.GRAPH.canonicalize(
                    f'{h.graph_key}@{h.graph_data_version}')
                for h in handles
            }

        # The initial heartbeat is the only place that needs retries:
        # transient ``RpcError`` (controller-restart in flight at worker
        # startup, k8s rolling deploy of the controller pod) is common
        # at boot, while during steady-state the periodic heartbeat
        # tick is the backstop. Retry up to 10 minutes; after that,
        # raise so worker init fails loudly rather than silently never
        # registering with the controller.
        async for attempt in tenacity.AsyncRetrying(
                retry=tenacity.retry_if_exception_type(_RpcError),
                stop=tenacity.stop_after_delay(
                    timedelta(minutes=10).total_seconds()),
                wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),
                before_sleep=tenacity.before_sleep_log(log, _logging.WARNING),
                reraise=True,
        ):
            with attempt:
                await self._send_status_report()
        if self._storage_mode.is_long_lived:
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(), name='pool-heartbeat')

    async def stop(self) -> None:
        """Cancel background tasks and wait for them to exit. Idempotent."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
        # Close the job-output StateDB the store opened on first write (no-op
        # if a pure-ingest daemon never wrote). Best-effort: process exit would
        # reclaim it anyway.
        if self._job_output_store is not None:
            await self._job_output_store.aclose()

    def _get_job_output_store(self) -> MLJobOutputStore:
        """Return the self-contained store, creating it on first use.

        Construction does no I/O; the store opens one StateDB on its first
        write and reuses it (``stop`` closes it). Both callers (``train``
        progress and the ``materialize_prediction_task`` wrapper's
        "materializing" write) run on the worker's event loop, so every write
        -- and the store's one lazy StateDB open -- happens on that one loop,
        which the reused connection pool is bound to. Racing first-callers
        (e.g. ``gpu_concurrency`` > 1) would at worst build two inert store
        objects here; the store's own lock guards its lazy StateDB open.
        """
        if self._job_output_store is None:
            self._job_output_store = MLJobOutputStore(self_contained=True)
        return self._job_output_store

    @property
    def activity_done(self) -> asyncio.Event:
        """Set after the first activity reaches a terminal state when
        ``storage_mode.is_activity_scoped``. The worker entry point awaits this
        and triggers a graceful Temporal worker shutdown so the
        surrounding harness (K8s Job, subprocess fixture, etc.) can
        reclaim the process.

        Long-lived workers never set this — they stop only when the
        process is killed externally.
        """
        return self._activity_done

    @property
    def activity_failed(self) -> bool:
        """True if the most recent activity body raised a retryable
        failure. Read by the worker entry point on activity-scoped modes to
        decide between exit-0 (success / non-retryable / cancellation)
        and exit-1 (retryable failure → K8s OnFailure restart →
        Temporal retry attempt onto fresh pod). See ``_activity_scope``
        for the classification.
        """
        return self._activity_failed

    # -----------------------------------------------------------------
    # Heartbeat / status reporting.
    # -----------------------------------------------------------------

    def _harvest_disk_probe(self) -> Optional[_DiskTelemetry]:
        task = self._disk_probe_task
        if task is None or not task.done():
            return None

        try:
            telemetry = task.result()
        except asyncio.CancelledError:
            log.debug('Local disk probe task was cancelled; reporting unknown')
            telemetry = _UNKNOWN_DISK_TELEMETRY
        except Exception:
            # The synchronous helper is deliberately non-raising. Keep this
            # second boundary so optional telemetry can never fail startup or
            # an event-carrying report if that contract regresses.
            log.debug('Local disk probe task failed; reporting unknown',
                      exc_info=True)
            telemetry = _UNKNOWN_DISK_TELEMETRY

        finished_at = self._monotonic()
        if self._disk_probe_timeout_warned:
            elapsed = (finished_at - self._disk_probe_started_at
                       if self._disk_probe_started_at is not None else 0.0)
            log.info('Local disk probe completed after %.1fs', elapsed)
        self._disk_telemetry = telemetry
        # A late sample is considered fresh when it is harvested. Its age
        # cannot be known while the underlying filesystem call is blocked.
        self._disk_telemetry_at = finished_at
        self._disk_probe_task = None
        self._disk_probe_started_at = None
        return telemetry

    def _start_disk_probe(
        self,
        data_root: Optional[str],
        dataplane_root: Optional[str],
    ) -> asyncio.Task[_DiskTelemetry]:
        assert self._disk_probe_task is None
        task = asyncio.create_task(
            asyncio.to_thread(_probe_local_disks, data_root, dataplane_root),
            name='pool-disk-probe',
        )
        task.add_done_callback(_retrieve_task_exception)
        self._disk_probe_task = task
        self._disk_probe_started_at = self._monotonic()
        self._disk_probe_timeout_warned = False
        return task

    async def _resolve_disk_telemetry(
        self,
        data_root: Optional[str],
        dataplane_root: Optional[str],
    ) -> _DiskTelemetry:
        self._harvest_disk_probe()
        task = self._disk_probe_task
        if task is None:
            task = self._start_disk_probe(data_root, dataplane_root)

        now = self._monotonic()
        if (self._disk_telemetry_at is not None and
                now - self._disk_telemetry_at < _DISK_TELEMETRY_MAX_AGE_SECS):
            return self._disk_telemetry

        try:
            await asyncio.wait_for(asyncio.shield(task),
                                   timeout=_DISK_PROBE_WAIT_SECS)
        except TimeoutError:
            if not self._disk_probe_timeout_warned:
                elapsed = (self._monotonic() - self._disk_probe_started_at
                           if self._disk_probe_started_at is not None else
                           _DISK_PROBE_WAIT_SECS)
                log.warning(
                    'Local disk probe has not completed after %.1fs; '
                    'reporting unknown capacity', elapsed)
                self._disk_probe_timeout_warned = True
            return _UNKNOWN_DISK_TELEMETRY
        except asyncio.CancelledError:
            # Shielding preserves the probe when the report is cancelled. A
            # cancellation of the retained task itself is optional telemetry,
            # not a reason to cancel worker startup or status reporting.
            if not task.cancelled():
                raise
        except Exception:
            # ``wait_for`` propagates the retained task's exception. Harvest
            # it through the non-raising boundary below.
            pass

        harvested = self._harvest_disk_probe()
        return (harvested if harvested is not None else _UNKNOWN_DISK_TELEMETRY)

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat. Cancellation propagates; transient send
        failures are swallowed by ``_send_status_report`` itself (the
        controller's miss-threshold is the backstop)."""
        while True:
            await asyncio.sleep(POOL_HEARTBEAT_INTERVAL_SECS)
            await self._send_status_report(raise_on_error=False)

    async def _send_status_report(
        self,
        event: Optional[WorkerEvent] = None,
        *,
        raise_on_error: bool = True,
    ) -> None:
        """Emit a ``WorkerStatusReport`` (heartbeat or event-carrying).

        Populates ``ready_snapshots`` (the controller's WARM-vs-MISSING
        signal) and SSD totals from observed local state. GPU capacity and
        in-use counts are controller-owned: capacity comes from the persisted
        durable tier, and in-use slots come from approved leases. The daemon
        does not query K8s node capacity or report authoritative GPU slots.
        """
        async with self._status_report_lock:
            await self._send_status_report_locked(event,
                                                  raise_on_error=raise_on_error)

    async def _send_status_report_locked(
        self,
        event: Optional[WorkerEvent],
        *,
        raise_on_error: bool,
    ) -> None:
        from service_framework._raw_deploy_config import DiskgraphConfig
        config = deploy_env()
        dc = config.data_compute
        data_root = dc.data_root if isinstance(dc, DiskgraphConfig) else None

        ssd_probe, dataplane_probe = await self._resolve_disk_telemetry(
            data_root, config.data_plane_storage_root)
        ssd_total_bytes, ssd_used_bytes, ssd_device = ssd_probe
        dataplane_total_bytes, dataplane_used_bytes, dataplane_device = (
            dataplane_probe)
        same_filesystem = (ssd_device == dataplane_device
                           if ssd_device is not None
                           and dataplane_device is not None else None)

        # Only the daemon reports ``ready_snapshots`` — it owns the
        # PVC and is authoritative for what's WARM on this worker.
        # Activity-scoped pods (overflow emptyDir, or activity-pod
        # sharing the daemon's PVC) leave it empty so the controller
        # doesn't double-count or mistake transient emptyDir snapshots
        # for durably cached ones.
        is_long_lived = self._storage_mode.is_long_lived
        ready_snapshots = (sorted(self._ready_snapshots)
                           if is_long_lived else [])
        report = WorkerStatusReport(
            worker_name=self._uid,
            state=WorkerStateSnapshot(
                ssd_total_bytes=ssd_total_bytes,
                ssd_used_bytes=ssd_used_bytes,
                ready_snapshots=ready_snapshots,
                dataplane_total_bytes=dataplane_total_bytes,
                dataplane_used_bytes=dataplane_used_bytes,
                dataplane_same_filesystem_as_ssd=same_filesystem,
            ),
            worker_storage_mode=self._storage_mode,
            event=event,
        )
        try:
            await self._controller.report_status(report)
        except _RpcError:
            if raise_on_error:
                raise
            log.error('report_status failed (event=%s)',
                      event.type.value if event is not None else None,
                      exc_info=True)

    # -----------------------------------------------------------------
    # Activity-scoped (job-scoped) body wrapper — every job-scoped
    # @activity.defn method (gen_tt / materialize / train / predict)
    # enters this context. Ensures the activity-scoped self-exit signal fires
    # on any exit path (success, raised exception, cancellation) without
    # each activity having to write its own try/finally.
    #
    # Daemon-side activities (ensure_snapshot, delete_snapshot) do NOT
    # use this — they never run on an activity-scoped worker, so the self-exit
    # signal is meaningless for them.
    # -----------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _activity_scope(self) -> AsyncIterator[None]:
        """Wraps every job-scoped activity body.  On any exit fires the
        single-activity self-exit signal; tracks success/failure so the
        worker entrypoint can propagate non-zero exit on failure (per
        the activity-scoped pod retry contract — k8s Job ``OnFailure``
        restart + Temporal reschedules the retry attempt).

        ``CancelledError`` is treated as a retryable failure here, on
        purpose. Two cancellation sources land in this branch:

        * **Heartbeat-timeout cancellation** — Temporal Cloud sees
          missed heartbeats, fails the attempt with cancellation, and
          schedules a retry. By the time the pod recovers (e.g.,
          finishing a CPU/IO-intensive work) and observes the cancel,
          a *new* attempt is queued. Exiting non-zero → K8s restart →
          fresh pod picks up the retry. Exiting 0 here would leave the
          retry attempt with no worker.
        * **Workflow cancellation** — caller no longer wants the work.
          A restarted pod has nothing to do. Pod-side dangling is
          accepted as the cost of correctness for the heartbeat case;
          the controller-side workflow-cancellation path is responsible
          for deleting the K8s Job/pod entirely (out of scope for
          ``_activity_scope``).

        ``is_error_non_retryable(ex) == True`` (ValueError, KeyError,
        AssertionError, ApplicationError(non_retryable=True), etc.) is
        the only branch that exits 0: Temporal classifies the activity
        as terminally failed; no retry attempt is scheduled, and a
        restarted pod has no task to pick up.
        """
        from temporalio_commons.error_utils import is_error_non_retryable
        try:
            yield
        except BaseException as ex:
            if not is_error_non_retryable(ex):
                self._activity_failed = True
            raise
        finally:
            if self._storage_mode.is_activity_scoped:
                self._activity_done.set()

    async def _ensure_snapshot(
        self,
        snapshot_id: SnapshotID,
        *,
        ingest_args: Optional[DiskgraphIngestArgs] = None,
    ) -> Optional[DiskgraphIngestResult]:
        """Inline-write when this worker owns its storage, delegate to
        the daemon when it doesn't.  See diskgraph_pool_controller.md
        § Snapshot Mutation Invariant.

        No ``@auto_heartbeater`` here on purpose. ``_ensure_snapshot`` is
        only called from inside ``@activity.defn`` methods (gen_tt /
        materialize_* / train / predict) whose own ``@auto_heartbeater``
        is already running on the outer activity. A nested
        ``_async_heartbeat_wrapper`` would create an independent
        ``asyncio.ensure_future(_ensure_snapshot_local(...))`` task that
        is NOT cancelled when the outer activity is cancelled — the
        inner orphan keeps running with the per-snapshot lock held for
        the full ingest duration.
        """
        if self._temporal_client is None:
            return await self._ensure_snapshot_local(snapshot_id,
                                                     ingest_args=ingest_args)
        assert ingest_args is None, (
            'activity-pod must never run daemon-mode (ingest_args set) '
            'ensure_snapshot — DiskgraphIngestWorkflow dispatches that '
            'directly to the daemon')
        info = activity.info()
        client = await self._temporal_client.get()
        # ID is stable across retries of the parent activity. If the
        # parent dies (pod eviction, OOM, schedule_to_close) while the
        # daemon's delegation is still running, ``USE_EXISTING`` lets
        # the next retry join that in-flight workflow rather than hit
        # ``WorkflowAlreadyStartedError``. Default ``id_reuse_policy``
        # (ALLOW_DUPLICATE) handles the already-closed case.
        await client.execute_workflow(
            RunSingleActivityWorkflow.run,
            args=[
                SnapshotActivities.ensure_snapshot.__name__,
                [snapshot_id],
            ],
            id=(f'ensure-snapshot:{snapshot_id}:{info.workflow_id}:'
                f'{info.activity_id}'),
            task_queue=daemon_dispatch_queue(self._uid),
            execution_timeout=_DAEMON_ENSURE_SNAPSHOT_TIMEOUT,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
        return None

    async def _ensure_snapshot_local(
        self,
        snapshot_id: SnapshotID,
        *,
        ingest_args: Optional[DiskgraphIngestArgs] = None,
    ) -> DiskgraphIngestResult:
        """Make ``snapshot_id`` resident on the local SSD.

        Two caller modes, distinguished by the presence of
        ``ingest_args``:

        * **Daemon mode** (``ingest_args=DiskgraphIngestArgs(...)``):
          SSD hit → dataplane restore → fresh ingest as a last resort.
          The daemon owns the ingest contract and is allowed to pay
          the cost.
        * **Consumer mode** (``ingest_args=None``): SSD hit → dataplane
          restore → raise ``ApplicationError`` with
          ``type=PERMANENT_SNAPSHOT_MISS_ERROR_TYPE`` and
          ``non_retryable=True``.
          Consumers (``gen_tt`` / ``materialize`` / ``train`` /
          ``predict``) cannot pay ingest cost; if both checks miss,
          the controller is expected to re-place this snapshot via a
          fresh daemon dispatch and the consumer activity will retry
          on the next dispatch.

        ``ingest_args`` is the typed bundle of everything the Rust
        ingest needs (see :class:`DiskgraphIngestArgs`).

        Per-snapshot lock serializes concurrent residency operations so
        duplicate work collapses.
        """
        # Lazy-import: only a worker executing snapshot operations needs to
        # load the native DiskGraph runtime.
        from diskgraph import DiskgraphCorruptionError

        # Canonicalize *before* acquiring the lock so callers passing
        # equivalent-but-not-byte-identical snapshot strings still
        # collapse onto the same lock; parse failures also surface
        # without holding any lock.
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        graph_fpid, graph_data_version = DataSnapshot.GRAPH.parse_id(
            snapshot_id)
        # The diskgraph runtime keys snapshots by the v2 ``GraphIDv2``
        # form (``"NA-<fpid>"`` for snapshot-attached graphs — the only
        # graph form the diskgraph pool serves), not the bare fpid.
        handle = DiskgraphHandle(
            graph_key=f'NA-{graph_fpid}',
            graph_data_version=graph_data_version,
        )
        service = get_service()
        dataplane_root = get_persist_dir()
        local_dir = service.local_snapshot_path(handle)
        # Every off-thread dispatch below preserves Temporal activity, tracing,
        # and logging contextvars. This is also required by the remaining
        # native operations that use process isolation.
        async with self._snapshot_locks.acquire(snapshot_id):
            resume_incomplete = False
            reuse_persisted_config = False
            # 1. SSD residency check. Two distinct miss kinds, logged
            # separately:
            #
            # 1a. *Clean miss* — ``local_dir`` doesn't exist. Common
            #     case (worker has never seen this snapshot, or it was
            #     evicted by retention). Fall through silently to the
            #     dataplane restore step.
            # 1b. *Corrupt cache* — ``local_dir`` exists but
            #     ``verify_integrity`` raises
            #     :class:`DiskgraphCorruptionError` (missing
            #     ``_SUCCESS`` / ``_MANIFEST.json``, manifest version
            #     mismatch, size mismatch, partial files from a
            #     previous interrupted ingest, etc.). Worth alerting
            #     on — surfaces silent SSD corruption / aborted
            #     ingests. Rmtree the dir before falling through so
            #     the next step (fsutil.copy from dataplane, or fresh
            #     ingest) doesn't get tangled with leftover files.
            with kumo_timing("snapshot.ensure.ssd_check"):
                if not await asyncio.to_thread(os.path.isdir, local_dir):
                    log.info(
                        '_ensure_snapshot: %s SSD miss (no local dir at %s)',
                        snapshot_id, local_dir)
                    reuse_persisted_config = (await _run_diskgraph_async(
                        service.has_persisted_ingest_config,
                        handle,
                    )) is True
                    self._ready_snapshots.discard(snapshot_id)
                    await _run_diskgraph_async(
                        service.cleanup_ingest_sidecars,
                        handle,
                        preserve_config=reuse_persisted_config,
                    )
                else:
                    try:
                        resume_incomplete = await _run_diskgraph_async(
                            service.has_resumable_ingest, handle)
                        if resume_incomplete:
                            log.info(
                                '_ensure_snapshot: %s has a durable Encode '
                                'marker; preserving it for resume', snapshot_id)
                            self._ready_snapshots.discard(snapshot_id)
                            cached = None
                        else:
                            cached = await _run_diskgraph_async(
                                service.verify_integrity, handle)
                        if cached is not None:
                            log.info(
                                '_ensure_snapshot: %s SSD hit (integrity OK)',
                                snapshot_id)
                            size_bytes = await _snapshot_dir_size_bytes(
                                snapshot_id, local_dir)
                            self._ready_snapshots.add(snapshot_id)
                            if self._storage_mode.is_long_lived:
                                await self._send_status_report(
                                    event=WorkerEvent(
                                        type=WorkerEventType.SNAPSHOT_READY,
                                        snapshot_id=snapshot_id,
                                        size_bytes=size_bytes),
                                    raise_on_error=False)
                            return cached
                    except DiskgraphCorruptionError as exc:
                        log.warning(
                            '_ensure_snapshot: %s SSD cache CORRUPT at %s '
                            '(%s); invalidating and cleaning up', snapshot_id,
                            local_dir, exc)
                        reuse_persisted_config = (await _run_diskgraph_async(
                            service.has_persisted_ingest_config,
                            handle,
                        )) is True
                        self._ready_snapshots.discard(snapshot_id)
                        await self._send_status_report(
                            event=WorkerEvent(
                                type=WorkerEventType.SSD_CACHE_CORRUPTED,
                                snapshot_id=snapshot_id,
                            ),
                            raise_on_error=False,
                        )
                        try:
                            try:
                                await asyncio.to_thread(shutil.rmtree,
                                                        local_dir)
                            except FileNotFoundError:
                                pass
                            await _run_diskgraph_async(
                                service.cleanup_ingest_sidecars,
                                handle,
                                preserve_config=reuse_persisted_config,
                            )
                        except Exception as cleanup_exc:
                            raise exc from cleanup_exc

            # 2. Dataplane restore. ``service.restore_snapshot`` does
            # the parallel copy + manifest cross-check; returns the
            # ``IngestResult`` on hit and ``None`` when no mirror exists.
            # Any exception while copying/checking is ambiguous: transfer or
            # local-disk failure looks the same as bad remote bytes here.
            # Clean only the incomplete local target and let the existing
            # bounded activity/workflow policy retry. L2 remains trusted.
            dataplane_dir = service.dataplane_snapshot_path(
                handle, dataplane_root)
            restored = None
            if not resume_incomplete:
                log.info('_ensure_snapshot: %s attempting dataplane restore',
                         snapshot_id)
                await self._send_status_report(
                    event=WorkerEvent(
                        type=WorkerEventType.SNAPSHOT_RESTORE_STARTED,
                        snapshot_id=snapshot_id), raise_on_error=False)
                activity_loop = asyncio.get_running_loop()

                def restore_heartbeat() -> None:
                    _schedule_activity_heartbeat(activity_loop)

                try:
                    with kumo_timing("snapshot.ensure.dataplane_restore"):
                        restored = await _run_diskgraph_async(
                            service.restore_snapshot, handle, dataplane_root,
                            heartbeat=restore_heartbeat,
                            # ``to_thread`` cannot stop ``fsutil.copy`` while
                            # it is writing into ``local_dir``. Keep the lock
                            # until that writer exits, then remove its
                            # partial tree.
                            join_on_cancel=True,
                            cancellation_cleanup_dir=local_dir)
                except Exception as exc:
                    log.warning(
                        '_ensure_snapshot: %s dataplane restore failed from %s '
                        'into %s (%s); cleaning only the partial local copy',
                        snapshot_id, dataplane_dir, local_dir, exc)
                    await asyncio.to_thread(shutil.rmtree, local_dir,
                                            ignore_errors=True)
                    await _run_diskgraph_async(
                        service.cleanup_ingest_sidecars,
                        handle,
                        preserve_config=reuse_persisted_config,
                    )
                    raise RuntimeError(
                        f'failed to restore snapshot {snapshot_id} from '
                        f'dataplane {dataplane_dir} into local {local_dir}: '
                        f'{exc}') from exc

            if restored is not None:
                log.info('_ensure_snapshot: %s restored from dataplane',
                         snapshot_id)
                if reuse_persisted_config:
                    await _run_diskgraph_async(
                        service.cleanup_ingest_sidecars,
                        handle,
                    )
                size_bytes = await _snapshot_dir_size_bytes(
                    snapshot_id, local_dir)
                self._ready_snapshots.add(snapshot_id)
                await self._send_status_report(
                    event=WorkerEvent(
                        type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
                        snapshot_id=snapshot_id, size_bytes=size_bytes,
                        dataplane_path=dataplane_dir), raise_on_error=False)
                await self._send_status_report(
                    event=WorkerEvent(type=WorkerEventType.SNAPSHOT_READY,
                                      snapshot_id=snapshot_id,
                                      size_bytes=size_bytes),
                    raise_on_error=False)
                return restored
            if not resume_incomplete:
                log.info(
                    '_ensure_snapshot: %s no dataplane mirror; falling '
                    'through to fresh ingest', snapshot_id)

            # 3. Neither SSD nor dataplane has the snapshot.
            if ingest_args is None:
                # Raise as an explicitly non-retryable ApplicationError.
                # A Python ``ValueError`` subclass is *not* picked up
                # by Temporal's ``non_retryable_error_types`` matching
                # (it compares class names, not MRO), so consumers
                # would retry in a tight loop. ``non_retryable=True``
                # short-circuits that.
                raise ApplicationError(
                    f'snapshot {snapshot_id} missing on this pool '
                    f'worker (no SSD cache, no dataplane backup) and '
                    f'caller passed ingest_args=None (consumer mode). '
                    f'Activity is permanently failed against this '
                    f'worker — the controller must re-place the '
                    f'snapshot via a fresh daemon ensure_snapshot '
                    f'dispatch before the consumer can retry.',
                    type=PERMANENT_SNAPSHOT_MISS_ERROR_TYPE,
                    non_retryable=True,
                )

            # 4. Fresh ingest. Rust streams the snapshot output to the
            # dataplane mirror as it writes, finalizes
            # ``_MANIFEST.json``, and writes ``_SUCCESS`` last on both
            # backends — the call returns only when both are durable.
            # No separate ``write_success_marker`` or post-ingest copy
            # needed.
            #
            # The ingest semaphore caps concurrent fresh ingests across
            # different snapshots on this worker (CPU + NVMe bound).
            # Acquired AFTER the per-snapshot lock so same-snapshot
            # callers serialize on the cheaper lock first.
            log.info(
                '_ensure_snapshot: %s SSD miss + dataplane miss; fresh '
                'ingest with streaming dataplane mirror', snapshot_id)
            await self._send_status_report(
                event=WorkerEvent(type=WorkerEventType.SNAPSHOT_INGEST_STARTED,
                                  snapshot_id=snapshot_id),
                raise_on_error=False)
            subprocess_environment: dict[str, Optional[str]] = {}
            try:
                if resume_incomplete or reuse_persisted_config:
                    source_specs = {}
                    source_paths = {}
                    graph_config = None
                else:
                    source_specs = await _run_diskgraph_async(
                        service.prepare_direct_sources,
                        connector_specs=ingest_args.connector_specs,
                        all_graph_tables=ingest_args.all_graph_tables,
                        data_persist_dir=ingest_args.data_persist_dir,
                        table_data_versions=ingest_args.table_data_versions,
                        source_size_bytes_by_table=(
                            ingest_args.source_size_bytes_by_table),
                    )
                    source_paths = {
                        alias:
                        get_diskgraph_snapshot_ingest_path(
                            snapshot_id, table.fingerprint(),
                            str(ingest_args.table_data_versions[alias]))
                        for alias, table in
                        ingest_args.all_graph_tables.items()
                        if alias not in source_specs
                    }
                    # Only staged/ineligible file sources are downloaded
                    # before taking an Encode slot. Direct warehouse/object
                    # sources have no staged file or marker.
                    if source_paths:
                        with kumo_timing(
                                "snapshot.ensure.download_table_files",
                                table_count=len(source_paths),
                        ):
                            await _run_diskgraph_async(
                                service.download_table_files,
                                connector_specs=ingest_args.connector_specs,
                                table_data_versions=(
                                    ingest_args.table_data_versions),
                                all_graph_tables=(ingest_args.all_graph_tables),
                                source_paths=source_paths,
                                # ``to_thread`` cannot stop a connector that
                                # is already writing.  Remove the whole
                                # snapshot tree after that writer exits so it
                                # cannot recreate data behind workflow cleanup.
                                join_on_cancel=True,
                                cancellation_cleanup_dir=(
                                    get_diskgraph_snapshot_ingest_dir(
                                        snapshot_id)),
                            )
                    graph_config = await asyncio.to_thread(
                        _build_diskgraph_ingest_config, ingest_args)
                subprocess_environment = await _run_diskgraph_async(
                    service.prepare_subprocess_environment,
                    connector_specs=ingest_args.connector_specs,
                    all_graph_tables=ingest_args.all_graph_tables,
                    source_specs=(None if resume_incomplete
                                  or reuse_persisted_config else source_specs),
                    data_persist_dir=ingest_args.data_persist_dir,
                    table_data_versions=ingest_args.table_data_versions,
                )

                activity_loop = asyncio.get_running_loop()

                def heartbeat() -> None:
                    _schedule_activity_heartbeat(activity_loop)

                async with self._ingest_semaphore:
                    with kumo_timing(
                            "snapshot.ensure.diskgraph_ingest",
                            table_count=len(ingest_args.all_graph_tables),
                    ):
                        try:
                            result = await _run_diskgraph_async(
                                service.ingest,
                                graph_config=graph_config,
                                handle=handle,
                                data_persist_dir=ingest_args.data_persist_dir,
                                table_data_versions=(
                                    ingest_args.table_data_versions),
                                source_paths=(source_paths or None),
                                source_specs=(source_specs or None),
                                subprocess_environment=(subprocess_environment
                                                        or None),
                                heartbeat=heartbeat,
                                dataplane_root=dataplane_root,
                                join_on_cancel=True,
                            )
                        except Exception as exc:
                            if not isinstance(exc, DiskgraphNoSpaceError):
                                raise
                            # A larger-tier retry must start from zero. Keeping
                            # Encode state would bind it to the failed local
                            # disk, and keeping a partial dataplane mirror
                            # would mix attempts from different workers.
                            self._ready_snapshots.discard(snapshot_id)
                            await asyncio.to_thread(shutil.rmtree, local_dir,
                                                    ignore_errors=True)
                            await asyncio.to_thread(fsutil.rm, dataplane_dir,
                                                    ignore_errors=True)
                            await _run_diskgraph_async(
                                service.cleanup_ingest_sidecars, handle)
                            raise
            finally:
                subprocess_environment.clear()
            await _run_diskgraph_async(service.cleanup_ingest_sidecars, handle)
            with kumo_timing("snapshot.ensure.snapshot_dir_size"):
                size_bytes = await _snapshot_dir_size_bytes(
                    snapshot_id, local_dir)
            self._ready_snapshots.add(snapshot_id)
            await self._send_status_report(
                event=WorkerEvent(
                    type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
                    snapshot_id=snapshot_id, size_bytes=size_bytes,
                    dataplane_path=dataplane_dir), raise_on_error=False)
            await self._send_status_report(
                event=WorkerEvent(type=WorkerEventType.SNAPSHOT_READY,
                                  snapshot_id=snapshot_id,
                                  size_bytes=size_bytes), raise_on_error=False)
            return result

    # -----------------------------------------------------------------
    # Daemon-style activities (registered on workers that own the PVC —
    # K8s daemon pod, singlehost, devserver, etc.)
    # -----------------------------------------------------------------

    @connector_activity_registry.connector_authorized_activity
    async def ensure_snapshot(
        self,
        snapshot_id: SnapshotID,
        ingest_args: Optional[DiskgraphIngestArgs] = None,
    ) -> Optional[DiskgraphIngestResult]:
        """Daemon-dispatched: make ``snapshot_id`` resident on local SSD.

        Bypasses the :meth:`_ensure_snapshot` dispatcher and goes
        straight to :meth:`_ensure_snapshot_local` because this activity
        IS the writer for the daemon's PVC.

        Two callers: ``DiskgraphIngestWorkflow`` (with ``ingest_args``,
        returns ``DiskgraphIngestResult``), and the activity-pod-side
        delegation in :meth:`_ensure_snapshot` (without, returns ``None``).
        """
        return await self._ensure_snapshot_local(snapshot_id,
                                                 ingest_args=ingest_args)

    @activity.defn
    async def delete_snapshot(self, snapshot_id: SnapshotID) -> None:
        """Evict ``snapshot_id`` from this worker's PVC.

        Called by the controller when retention policy fires (no leases
        + idle window expired). Idempotent: deleting an already-absent
        snapshot is a no-op.

        Daemon-side; never runs on an activity-scoped worker, so no
        ``_activity_scope`` wrap.

        The controller observes the next heartbeat/status report and
        reconciles local inventory from ``ready_snapshots``.
        """
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        async with self._snapshot_locks.acquire(snapshot_id):
            graph_fpid, graph_data_version = DataSnapshot.GRAPH.parse_id(
                snapshot_id)
            handle = DiskgraphHandle(
                graph_key=f'NA-{graph_fpid}',
                graph_data_version=graph_data_version,
            )
            local_dir = get_service().local_snapshot_path(handle)
            await asyncio.to_thread(shutil.rmtree, local_dir,
                                    ignore_errors=True)
            await _run_diskgraph_async(get_service().cleanup_ingest_sidecars,
                                       handle)
            self._ready_snapshots.discard(snapshot_id)

        await self._send_status_report(raise_on_error=False)

    # -----------------------------------------------------------------
    # Activity-scoped methods (registered on workers that run job-scoped
    # compute — K8s activity-scoped pod, singlehost, CI test process,
    # etc.). On activity-scoped workers, the bootstrap typically registers
    # exactly one of these (the activity this worker exists to run).
    # -----------------------------------------------------------------

    @connector_activity_registry.connector_authorized_activity
    async def gen_tt(
        self,
        snapshot_id: SnapshotID,
        job_input: Any,
        settings: Any,
        home_dir: Optional[str] = None,
        is_child_job: bool = False,
    ) -> Tuple[str, Any, str]:
        """Thin wrapper around ``generate_target_table`` —
        ``generate_training_table`` / ``generate_prediction_table``
        share the same body, picked here by ``job_input`` type."""
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        async with self._activity_scope():
            await self._ensure_snapshot(snapshot_id)
            if isinstance(job_input, BatchPredictionJobInput):
                return await _run_diskgraph_async(generate_prediction_table,
                                                  job_input, settings, home_dir,
                                                  is_child_job)
            return await _run_diskgraph_async(generate_training_table,
                                              job_input, settings)

    @connector_activity_registry.connector_authorized_activity
    async def build_ttng_custom_train_table_bundle(
        self,
        snapshot_id: SnapshotID,
        job_input: Any,
        settings: Any,
        train_table_path: str,
        weight_col_name: Optional[str] = None,
    ) -> str:
        """Thin wrapper around ``build_ttng_custom_train_table_bundle``.

        The custom-TT gen wheel opens the PARENT graph index at
        ``<data_root>/<graph>/<ver>/index_metadata.json`` to compute the
        cross-boundary FK layout. On the [s3] lane that parent index is only
        resident on a worker after the pool restores the snapshot, so route
        this activity through the pool (like ``gen_tt``) and
        ``_ensure_snapshot`` before delegating -- same residency guarantee the
        normal materialize/gen path relies on."""
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        async with self._activity_scope():
            await self._ensure_snapshot(snapshot_id)
            return await _run_diskgraph_async(
                build_ttng_custom_train_table_bundle,
                job_input,
                settings,
                train_table_path,
                weight_col_name,
            )

    @connector_activity_registry.connector_authorized_activity
    async def build_ttng_custom_pred_table_bundle(
        self,
        snapshot_id: SnapshotID,
        job_input: Any,
        settings: Any,
        pred_table_path: str,
    ) -> None:
        """Build a custom PRED bundle beside its resident graph snapshot."""
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        async with self._activity_scope():
            await self._ensure_snapshot(snapshot_id)
            return await _run_diskgraph_async(
                build_ttng_custom_pred_table_bundle,
                job_input,
                settings,
                pred_table_path,
            )

    @connector_activity_registry.connector_authorized_activity
    async def materialize_training_task(
        self,
        snapshot_id: SnapshotID,
        job_input: Any,
        settings: Any,
        train_table_path: str,
        train_table_materialization_path: str,
        job_config_path: str,
        weight_col_name: Optional[str] = None,
    ) -> Tuple[str, str, Any]:
        """Thin wrapper around ``materialize_training_task``."""
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        async with self._activity_scope():
            await self._ensure_snapshot(snapshot_id)
            result = await _run_diskgraph_async(
                materialize_training_task,
                job_input,
                settings,
                train_table_path,
                train_table_materialization_path,
                job_config_path,
                weight_col_name,
            )
            await asyncio.to_thread(
                upload_job_sidecars_from_config_path,
                snapshot_id,
                result[0],
                prediction=False,
                data_persist_dir=get_persist_dir(),
            )
            return result

    @connector_activity_registry.connector_authorized_activity
    async def materialize_prediction_task(
        self,
        snapshot_id: SnapshotID,
        job_input: Any,
        settings: Any,
        serving: bool,
        pred_config_path: str,
        home_dir: Optional[str] = None,
        job_type: Any = None,
    ) -> Tuple[str, str, Any]:
        """Thin wrapper around ``materialize_prediction_task``."""
        if job_type is None:
            job_type = JobType.BATCH_PREDICTION_JOB
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        async with self._activity_scope():
            await self._ensure_snapshot(snapshot_id)
            # Publish the "materializing" status here, on the worker's event
            # loop, rather than from inside the diskgraph worker thread (where
            # it previously ran via ``AsyncToSync``). This keeps every
            # job-output StateDB write on the same loop the store is reused on.
            await update_job_status(job_input.id, job_type,
                                    self._get_job_output_store(),
                                    JobEventLogEntry.materializing())
            return await _run_diskgraph_async(
                _materialize_and_upload_prediction_task,
                snapshot_id,
                job_input,
                settings,
                serving,
                pred_config_path,
                home_dir,
                get_persist_dir(),
                join_on_cancel=True,
            )

    @connector_activity_registry.connector_authorized_activity
    async def train(self, snapshot_id: SnapshotID, job_id: JobID,
                    job_config_path: str) -> JobState:
        """Run GNN training in a kumo-auto subprocess."""
        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        # ``_activity_scope`` outer so its ``finally`` (which sets
        # ``activity_done`` for activity-scoped self-exit) fires even on
        # cancellation while waiting for the GPU semaphore.
        async with self._activity_scope():
            async with self._gpu_slots.acquire() as gpu_environment:
                with kumo_timing(
                        "pool.train.ensure_snapshot",
                        activity="train",
                ):
                    await self._ensure_snapshot(snapshot_id)
                use_ttng_sidecars = True  # TT-NG is the only mode; sidecar hand-off always on
                sidecars = None
                try:
                    if use_ttng_sidecars:
                        # Sidecar hand-off is authoritative: materialize
                        # uploaded the per-job train_tables extension, so a
                        # restore failure is a real bug -- let it propagate
                        # (fail loud). Temporal retries transient failures.
                        with kumo_timing(
                                "pool.train.restore_sidecars",
                                activity="train",
                        ):
                            sidecars = await asyncio.to_thread(
                                restore_job_sidecars_for_use_from_config_path,
                                snapshot_id,
                                job_config_path,
                                prediction=False,
                                data_persist_dir=get_persist_dir(),
                            )
                    with kumo_timing(
                            "pool.train.subprocess",
                            activity="train",
                            binary="kumo-auto",
                    ):
                        return await _run_kumo_subprocess(
                            'kumo-auto', job_id, job_config_path,
                            self._get_job_output_store(),
                            **({
                                'subprocess_environment': gpu_environment
                            } if gpu_environment else {}))
                finally:
                    if use_ttng_sidecars and sidecars is not None:
                        with kumo_timing(
                                "pool.train.cleanup_sidecars",
                                activity="train",
                        ):
                            await asyncio.to_thread(sidecars.cleanup)

    @connector_activity_registry.connector_authorized_activity
    async def predict(
        self,
        snapshot_id: SnapshotID,
        job_id: JobID,
        job_config_path: str,
        shard_index: int = 0,
        num_shards: int = 1,
        job_input: BatchPredictionJobInput | None = None,
        artifact_ownership_enabled: bool = False,
        producer_stats_enabled: bool = False,
    ) -> PredictionShardResult:
        """Run batch prediction in a kumo-pred subprocess.

        Databricks-bound shards are written to local NVMe and uploaded from
        this GPU worker into a shared managed-volume prefix. Other output
        types retain the durable Kumo artifact path.
        """
        from kumo_core.job_config import PredictionConfig

        from kumo.train.artifact_manifest import (
            PREDICTION_ARTIFACT_MANIFEST_VERSION,
            PredictionShardManifest,
            hash_prediction_config,
            inventory_generation,
            prediction_shard_producer_abi,
            publish_shard_manifest,
        )
        from rest.backends.data_backend import connector_broker
        from rest.config.prediction_output import DatabricksPredictionOutput
        from statedb.config.common import ConnectorID

        snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        async with self._activity_scope():
            async with self._gpu_slots.acquire() as gpu_environment:
                outputs = (job_input.config.outputs
                           if job_input is not None else [])
                if artifact_ownership_enabled and job_input is None:
                    raise ValueError(
                        'Batch prediction ownership requires persisted job input'
                    )
                if producer_stats_enabled and not artifact_ownership_enabled:
                    raise ValueError(
                        'Prediction producer stats require artifact ownership')
                requested_artifacts = (tuple(
                    sorted(
                        requested_prediction_artifact_types(job_input.config),
                        key=lambda value: value.value,
                    )) if artifact_ownership_enabled else ())
                databricks_outputs = [
                    (index, output) for index, output in enumerate(outputs)
                    if isinstance(output, DatabricksPredictionOutput)
                ]
                needs_durable_artifact = (not outputs or any(
                    not isinstance(output, DatabricksPredictionOutput)
                    for output in outputs))
                original_config = await asyncio.to_thread(
                    PredictionConfig.load, job_config_path)
                original_output_dir = original_config.output_dir
                attempt_id = uuid.uuid4().hex
                source_config_sha256 = None
                if artifact_ownership_enabled:
                    source_config_sha256 = await asyncio.to_thread(
                        hash_prediction_config, job_config_path)
                    generation_output_dir = osp.join(
                        original_output_dir,
                        '_attempts',
                        f'runtime-{shard_index:05d}',
                        attempt_id,
                    )
                else:
                    generation_output_dir = original_output_dir
                local_output_dir = None
                subprocess_output_dir = generation_output_dir
                if databricks_outputs:
                    data_root = getattr(deploy_env().data_compute, 'data_root',
                                        None)
                    if data_root is not None:
                        os.makedirs(data_root, exist_ok=True)
                    local_output_dir = tempfile.mkdtemp(
                        prefix=f'kumo-pred-{job_id}-{shard_index:05d}-',
                        dir=data_root,
                    )
                    subprocess_output_dir = local_output_dir
                with kumo_timing(
                        "pool.predict.shard_config",
                        activity="predict",
                        shard_index=shard_index,
                        num_shards=num_shards,
                ):
                    sharded_config_path = await _run_diskgraph_async(
                        _shard_pred_config, job_config_path, shard_index,
                        num_shards, subprocess_output_dir,
                        attempt_id if artifact_ownership_enabled else None)
                runtime_config = await asyncio.to_thread(
                    PredictionConfig.load, sharded_config_path)
                runtime_config_sha256 = None
                if artifact_ownership_enabled:
                    runtime_config_sha256 = await asyncio.to_thread(
                        hash_prediction_config, sharded_config_path)
                artifact_stats_request = None
                artifact_stats_dir = None
                artifact_stats_path = None
                if producer_stats_enabled and needs_durable_artifact:
                    assert runtime_config_sha256 is not None
                    output_start, output_stop = (
                        prediction_output_partition_range(runtime_config))
                    artifact_stats_request = PredictionArtifactStatsRequest(
                        attempt_id=attempt_id,
                        job_id=str(job_id),
                        runtime_config_sha256=runtime_config_sha256,
                        output_dir=subprocess_output_dir,
                        runtime_shard_id=shard_index,
                        runtime_shard_count=num_shards,
                        output_partition_start=output_start,
                        output_partition_stop=output_stop,
                        expected_artifact_types=requested_artifacts,
                    )
                    artifact_stats_request.validate()
                    data_root = getattr(deploy_env().data_compute, 'data_root',
                                        None)
                    if data_root is not None and fsutil.is_local(data_root):
                        os.makedirs(data_root, exist_ok=True)
                    else:
                        data_root = None
                    artifact_stats_dir = tempfile.mkdtemp(
                        prefix=(f'kumo-pred-stats-{job_id}-{shard_index:05d}-'),
                        dir=data_root,
                    )
                    artifact_stats_path = osp.join(artifact_stats_dir,
                                                   'producer-stats.json')
                use_ttng_sidecars = True  # TT-NG is the only mode; sidecar hand-off always on
                sidecars = None
                completed_uploads: list[tuple[Any, DatabricksVolumeShard]] = []
                try:
                    with kumo_timing(
                            "pool.predict.ensure_snapshot",
                            activity="predict",
                            shard_index=shard_index,
                            num_shards=num_shards,
                    ):
                        await self._ensure_snapshot(snapshot_id)
                    if use_ttng_sidecars:
                        # Sidecar hand-off is authoritative: materialize
                        # uploaded this shard's predict-stage extension, so a
                        # restore failure is a real bug -- let it propagate
                        # (fail loud). Temporal retries transient failures.
                        with kumo_timing(
                                "pool.predict.restore_sidecars",
                                activity="predict",
                                shard_index=shard_index,
                                num_shards=num_shards,
                        ):
                            sidecars = await asyncio.to_thread(
                                restore_job_sidecars_for_use_from_config_path,
                                snapshot_id,
                                sharded_config_path,
                                prediction=True,
                                data_persist_dir=get_persist_dir(),
                            )
                    with kumo_timing(
                            "pool.predict.subprocess",
                            activity="predict",
                            binary="kumo-pred",
                            shard_index=shard_index,
                            num_shards=num_shards,
                    ):
                        # predict never publishes training progress (only the
                        # kumo-auto branch does), so it needs no store and must
                        # not trigger a StateDB open.
                        subprocess_environment = dict(gpu_environment)
                        if artifact_stats_request is not None:
                            assert artifact_stats_path is not None
                            subprocess_environment |= {
                                PREDICTION_ARTIFACT_STATS_REQUEST_ENV:
                                json.dumps(
                                    artifact_stats_request.to_dict(),
                                    sort_keys=True,
                                    separators=(',', ':'),
                                ),
                                PREDICTION_ARTIFACT_STATS_PATH_ENV:
                                artifact_stats_path,
                            }
                        job_state = await _run_kumo_subprocess(
                            'kumo-pred',
                            job_id,
                            sharded_config_path,
                            None,
                            **({
                                'subprocess_environment': subprocess_environment
                            } if subprocess_environment else {}),
                        )

                    artifact_stats_payload = None
                    if (job_state == JobState.SUCCEEDED
                            and artifact_stats_request is not None):
                        assert artifact_stats_path is not None
                        artifact_stats_payload = await asyncio.to_thread(
                            _load_prediction_artifact_stats_payload,
                            artifact_stats_path,
                            artifact_stats_request,
                        )

                    if (job_state == JobState.SUCCEEDED
                            and local_output_dir is not None):
                        assert job_input is not None
                        for output_index, output in databricks_outputs:
                            reference = job_input.output_connectors[ConnectorID(
                                output.connector_id)]
                            resolved = (
                                await
                                connector_broker.connector_credential_broker.
                                resolve_async(reference))
                            artifact_path = osp.join(
                                local_output_dir,
                                output.artifact_type.output_file_name,
                            )
                            with kumo_timing(
                                    "pool.predict.databricks_upload",
                                    activity="predict",
                                    shard_index=shard_index,
                                    num_shards=num_shards,
                                    output_index=output_index,
                            ):
                                try:
                                    upload = await _run_diskgraph_async(
                                        _upload_prediction_artifact_to_databricks,
                                        resolved,
                                        artifact_path=artifact_path,
                                        artifact_type=output.artifact_type,
                                        job_id=job_id,
                                        output_index=output_index,
                                        shard_index=shard_index,
                                        attempt_id=attempt_id,
                                        num_shards=num_shards,
                                        abandonment_cleanup_dir=local_output_dir,
                                    )
                                except asyncio.CancelledError:
                                    # The blocking SQL connector may still be
                                    # finishing a presigned PUT in its worker
                                    # thread. Its task owns deferred cleanup;
                                    # do not delete the source files here.
                                    local_output_dir = None
                                    raise
                            completed_uploads.append((resolved, upload))

                        if needs_durable_artifact:
                            await asyncio.to_thread(
                                _persist_prediction_artifacts,
                                local_output_dir,
                                generation_output_dir,
                            )

                    artifact_manifest = None
                    if (artifact_ownership_enabled
                            and job_state == JobState.SUCCEEDED
                            and needs_durable_artifact):
                        assert source_config_sha256 is not None
                        assert runtime_config_sha256 is not None
                        inventories = await asyncio.to_thread(
                            inventory_generation,
                            generation_output_dir,
                            requested_artifacts,
                            artifact_stats_payload,
                        )
                        output_start, output_stop = (
                            prediction_output_partition_range(runtime_config))
                        manifest = PredictionShardManifest(
                            schema_version=PREDICTION_ARTIFACT_MANIFEST_VERSION,
                            job_id=str(job_id),
                            attempt_id=attempt_id,
                            graph_snapshot_id=str(snapshot_id),
                            source_config_sha256=source_config_sha256,
                            runtime_config_sha256=runtime_config_sha256,
                            producer_abi=prediction_shard_producer_abi(
                                num_shards),
                            canonical_output_root=original_output_dir,
                            generation_root=generation_output_dir,
                            runtime_shard_id=shard_index,
                            runtime_shard_count=num_shards,
                            output_partition_start=output_start,
                            output_partition_stop=output_stop,
                            requested_artifacts=requested_artifacts,
                            artifacts=inventories,
                            producer_stats=producer_stats_enabled,
                        )
                        artifact_manifest = await asyncio.to_thread(
                            publish_shard_manifest, manifest)
                    elif (artifact_ownership_enabled
                          and job_state != JobState.SUCCEEDED):
                        await asyncio.to_thread(
                            fsutil.rm,
                            generation_output_dir,
                            ignore_errors=True,
                        )

                    return PredictionShardResult(
                        state=job_state,
                        databricks_uploads=[
                            upload for _, upload in completed_uploads
                        ],
                        artifact_manifest=artifact_manifest,
                    )
                except BaseException:
                    for connector_spec, upload in completed_uploads:
                        try:
                            await asyncio.to_thread(
                                _cleanup_databricks_prediction_upload,
                                connector_spec,
                                upload.staged_files,
                            )
                        except BaseException as cleanup_error:
                            log.warning(
                                'Databricks prediction upload rollback failed '
                                '(error_type=%s)',
                                type(cleanup_error).__name__,
                            )
                    if (artifact_ownership_enabled and not fsutil.exists(
                            osp.join(generation_output_dir,
                                     '_artifact_manifest.json'))):
                        try:
                            fsutil.rm(generation_output_dir, ignore_errors=True)
                        except BaseException:
                            log.warning(
                                'Prediction attempt cleanup failed '
                                '(job=%s shard=%s attempt=%s)',
                                job_id,
                                shard_index,
                                attempt_id,
                                exc_info=True,
                            )
                    raise
                finally:
                    async def _cleanup_prediction_attempt() -> None:
                        if use_ttng_sidecars and sidecars is not None:
                            with kumo_timing(
                                    "pool.predict.cleanup_sidecars",
                                    activity="predict",
                                    shard_index=shard_index,
                                    num_shards=num_shards,
                            ):
                                await asyncio.to_thread(sidecars.cleanup)
                        # Sharded configs are throwaway per-attempt; clean up
                        # so the job dir doesn't accumulate stale shards.
                        if sharded_config_path != job_config_path:
                            try:
                                os.remove(sharded_config_path)
                            except OSError:
                                log.debug(
                                    'pool predict: cleanup failed for sharded '
                                    'config %s (best-effort)',
                                    sharded_config_path)
                        if local_output_dir is not None:
                            await asyncio.to_thread(
                                shutil.rmtree,
                                local_output_dir,
                                ignore_errors=True,
                            )
                        if artifact_stats_dir is not None:
                            await asyncio.to_thread(
                                shutil.rmtree,
                                artifact_stats_dir,
                                ignore_errors=True,
                            )

                    cleanup_task = asyncio.create_task(
                        _cleanup_prediction_attempt(),
                        name=f'pool-predict-{job_id}-cleanup',
                    )
                    cleanup_cancellation = None
                    while not cleanup_task.done():
                        try:
                            await asyncio.shield(cleanup_task)
                        except asyncio.CancelledError as cancelled:
                            cleanup_cancellation = cancelled
                            log.warning(
                                'pool predict cleanup for job=%s shard=%s '
                                'was cancelled again; waiting for cleanup.',
                                job_id,
                                shard_index,
                            )
                    cleanup_task.result()
                    if cleanup_cancellation is not None:
                        raise cleanup_cancellation


def _shard_pred_config(
    orig_config_path: str,
    shard_index: int,
    num_shards: int,
    output_dir: str | None = None,
    attempt_id: str | None = None,
) -> str:
    """Write a shard-specific copy of ``orig_config_path``."""
    if num_shards <= 1 and output_dir is None:
        return orig_config_path
    from kumo_core.job_config import PredictionConfig

    from io_utils.json import load_json, save_json
    orig = load_json(orig_config_path, PredictionConfig)
    sharded = copy.deepcopy(orig)
    if num_shards <= 1:
        start, stop = orig.start_partition_id, orig.stop_partition_id
    elif orig.diskgraph_task_dataset_bundle_dir:
        start, stop = shard_index, shard_index + 1
        sharded.num_partitions = num_shards
    else:
        if orig.num_partitions < num_shards:
            raise ValueError(
                f'num_partitions={orig.num_partitions} < num_shards='
                f'{num_shards} for {orig_config_path}: the workflow must cap '
                f'num_shards at the materialized num_partitions')
        start, stop = split_partitions_across_jobs(orig.num_partitions,
                                                   num_shards)[shard_index]
    sharded.start_partition_id = start
    sharded.stop_partition_id = stop
    if output_dir is not None:
        sharded.output_dir = output_dir
        # Databricks recommends roughly 128 MiB compressed ingestion files.
        # This uncompressed bound is conservative and, critically, prevents
        # databricks-sql-connector from buffering multi-GiB PUT sources.
        sharded.max_prediction_file_size_bytes = (
            _DATABRICKS_PARQUET_FILE_TARGET_BYTES)
    base, ext = os.path.splitext(orig_config_path)
    attempt_suffix = f'-{attempt_id}' if attempt_id is not None else ''
    sharded_path = f'{base}-{start}-{stop}{attempt_suffix}{ext}'
    save_json(sharded, sharded_path)
    return sharded_path


# Heartbeat cadence while a long-running subprocess runs. Must be
# strictly less than the activity's ``heartbeat_timeout`` so Temporal
# doesn't mark the activity as failed mid-training.
_SUBPROCESS_HEARTBEAT_INTERVAL_SECS = 30
_SUBPROCESS_OUTPUT_CHUNK_BYTES = 64 << 10
_SUBPROCESS_OUTPUT_MAX_RECORD_BYTES = 64 << 10
_SUBPROCESS_PROGRESS_LOG_INTERVAL_SECS = 10.0
_SUBPROCESS_TERM_GRACE_SECS = 10.0
_SUBPROCESS_KILL_GRACE_SECS = 60.0
_SUBPROCESS_EXIT_POLL_SECS = 0.1
_SUBPROCESS_OWNERSHIP_HANDSHAKE_SECS = 10.0


@dataclasses.dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    process_group_id: int
    session_id: int
    start_time_ticks: int
    state: str
    proc_device: int
    proc_inode: int
    pid_namespace_device: int
    pid_namespace_inode: int
    task_states: tuple[tuple[int, str], ...] | None
    task_group_exited: bool | None = None

    @property
    def has_live_tasks(self) -> bool:
        """Whether any thread in this process has not completed exit."""
        if self.task_states is None:
            # pidfd polling is authoritative for the whole thread group. If
            # even that observation failed, retain the process conservatively.
            return self.task_group_exited is not True
        if not self.task_states:
            # A task can disappear between directory enumeration and its stat
            # read. A live leader is sufficient; for a zombie leader, use the
            # whole-thread-group pidfd observation and retain conservatively
            # if pidfd allocation or polling failed.
            return (self.state not in ('X', 'Z')
                    or self.task_group_exited is not True)
        return any(state not in ('X', 'Z') for _, state in self.task_states)


def _read_process_task_states(
    pid: int,
) -> tuple[tuple[int, str], ...] | None:
    """Read per-thread states so a zombie leader cannot hide live siblings."""
    task_states = []
    try:
        with os.scandir(f'/proc/{pid}/task') as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f'/proc/{pid}/task/{entry.name}/stat',
                              'rb') as stat_file:
                        task_stat = stat_file.read()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                fields = task_stat[task_stat.rfind(b')') + 2:].split()
                if fields:
                    task_states.append((int(entry.name), fields[0].decode()))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        # Linux documents this directory as unavailable after a thread-group
        # leader exits even while sibling threads can remain alive.
        return None
    return tuple(task_states)


def _read_process_identity(pid: int) -> _ProcessIdentity:
    """Read a PID identity that detects both reuse and namespace mismatch."""
    proc_path = f'/proc/{pid}'
    proc_before = os.stat(proc_path)
    pid_namespace = os.stat(f'{proc_path}/ns/pid')
    with open(f'{proc_path}/stat', 'rb') as stat_file:
        process_stat = stat_file.read()
    proc_after = os.stat(proc_path)
    if ((proc_before.st_dev, proc_before.st_ino)
            != (proc_after.st_dev, proc_after.st_ino)):
        raise ProcessLookupError(pid)

    # comm is parenthesized and may itself contain spaces or parentheses.
    fields = process_stat[process_stat.rfind(b')') + 2:].split()
    if len(fields) < 20:
        raise RuntimeError(f'Unexpected /proc/{pid}/stat contents')
    state = fields[0].decode()
    task_states = _read_process_task_states(pid)
    task_group_exited = None
    if (task_states is None or (not task_states and state in ('X', 'Z'))):
        try:
            task_group_exited = pidfd_process_exited(pid)
        except ProcessLookupError:
            raise
        except OSError:
            # Retain the identity conservatively if pidfd allocation/polling
            # is temporarily unavailable under resource pressure.
            pass
        proc_final = os.stat(proc_path)
        if ((proc_after.st_dev, proc_after.st_ino)
                != (proc_final.st_dev, proc_final.st_ino)):
            raise ProcessLookupError(pid)

    return _ProcessIdentity(
        pid=pid,
        state=state,
        process_group_id=int(fields[2]),
        session_id=int(fields[3]),
        start_time_ticks=int(fields[19]),
        proc_device=proc_after.st_dev,
        proc_inode=proc_after.st_ino,
        pid_namespace_device=pid_namespace.st_dev,
        pid_namespace_inode=pid_namespace.st_ino,
        task_states=task_states,
        task_group_exited=task_group_exited,
    )


@dataclasses.dataclass(frozen=True)
class _OwnedSubprocessSession:
    """A child-created session whose exact members are safe to signal.

    A descendant may create another process group without leaving this session,
    so ``killpg(leader.pid, ...)`` alone is not a complete production fence.
    Enumerating the session and signalling identity-checked pidfds covers those
    nested groups without selecting unrelated processes by a reusable bare PID.
    """
    leader: _ProcessIdentity

    @classmethod
    def capture(cls, pid: int) -> '_OwnedSubprocessSession':
        leader = _read_process_identity(pid)
        own_pid_namespace = os.stat('/proc/self/ns/pid')
        if ((leader.pid_namespace_device, leader.pid_namespace_inode)
                != (own_pid_namespace.st_dev, own_pid_namespace.st_ino)):
            raise RuntimeError(
                f'Subprocess pid={pid} is outside the worker PID namespace')
        if leader.process_group_id != pid or leader.session_id != pid:
            raise RuntimeError(
                f'Subprocess pid={pid} does not own its process session '
                f'(pgid={leader.process_group_id}, sid={leader.session_id})')
        return cls(leader)

    def members(self) -> list[_ProcessIdentity]:
        # If the leader PID has already been reused, the numeric session ID is
        # no longer proof of ownership. Refuse to enumerate or signal it.
        try:
            current_leader = _read_process_identity(self.leader.pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            current_leader = None
        if (current_leader is not None
                and not self._same_stable_identity(current_leader,
                                                   self.leader)):
            raise RuntimeError(
                f'Owned subprocess leader pid={self.leader.pid} was reused')

        members = []
        with os.scandir('/proc') as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    identity = _read_process_identity(int(entry.name))
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if (identity.session_id == self.leader.session_id
                        and identity.pid_namespace_device
                        == self.leader.pid_namespace_device
                        and identity.pid_namespace_inode
                        == self.leader.pid_namespace_inode):
                    members.append(identity)
        try:
            current_leader = _read_process_identity(self.leader.pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            current_leader = None
        if (current_leader is not None
                and not self._same_stable_identity(current_leader,
                                                   self.leader)):
            raise RuntimeError(
                f'Owned subprocess leader pid={self.leader.pid} was reused')
        return members

    def live_members(self) -> list[_ProcessIdentity]:
        return [member for member in self.members() if member.has_live_tasks]

    def reap_adopted_descendant_zombies(
            self, members: list[_ProcessIdentity]) -> int:
        """Reap exact session zombies only when this worker adopted them."""
        reaped = 0
        for member in members:
            if member.pid == self.leader.pid or member.has_live_tasks:
                continue
            if not self._identity_still_matches(member):
                continue
            try:
                waited_pid, _ = os.waitpid(member.pid, os.WNOHANG)
            except ChildProcessError:
                # A healthy supervisor remains this zombie's reaping parent.
                continue
            if waited_pid == member.pid:
                reaped += 1
        return reaped

    @staticmethod
    def _same_stable_identity(current: _ProcessIdentity,
                              expected: _ProcessIdentity) -> bool:
        return (current.pid == expected.pid
                and current.process_group_id == expected.process_group_id
                and current.session_id == expected.session_id
                and current.start_time_ticks == expected.start_time_ticks
                and current.proc_device == expected.proc_device
                and current.proc_inode == expected.proc_inode
                and current.pid_namespace_device
                == expected.pid_namespace_device
                and current.pid_namespace_inode
                == expected.pid_namespace_inode)

    @classmethod
    def _identity_still_matches(cls, expected: _ProcessIdentity) -> bool:
        try:
            current = _read_process_identity(expected.pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return False
        return cls._same_stable_identity(current, expected)

    def signal_members(self, members: list[_ProcessIdentity], sig: int) -> int:
        """Signal exact processes through pidfds, never a reusable bare PID."""
        signalled = 0
        for member in members:
            try:
                pidfd = pidfd_open(member.pid)
            except ProcessLookupError:
                continue
            try:
                if not self._identity_still_matches(member):
                    continue
                pidfd_send_signal(pidfd, sig)
                signalled += 1
            except ProcessLookupError:
                # The identity-checked process exited before pidfd delivery.
                pass
            finally:
                os.close(pidfd)
        return signalled

    async def terminate(self, binary_name: str, job_id: JobID) -> None:
        deadline = (asyncio.get_running_loop().time()
                    + _SUBPROCESS_TERM_GRACE_SECS)
        while True:
            members = self.members()
            self.reap_adopted_descendant_zombies(members)
            members = self.members()
            if not members:
                return
            self.signal_members(
                [member for member in members if member.has_live_tasks],
                signal.SIGTERM)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(_SUBPROCESS_EXIT_POLL_SECS, remaining))

        descendants = [
            member for member in members if member.pid != self.leader.pid
        ]
        log.warning(
            'pool %s subprocess session for job=%s still has %s descendant '
            'member(s) after %.1fs of SIGTERM; sending descendant SIGKILL.',
            binary_name, job_id, len(descendants),
            _SUBPROCESS_TERM_GRACE_SECS)
        deadline = (asyncio.get_running_loop().time()
                    + _SUBPROCESS_KILL_GRACE_SECS)
        while True:
            members = self.members()
            self.reap_adopted_descendant_zombies(members)
            members = self.members()
            if not members:
                return
            descendants = [
                member for member in members if member.pid != self.leader.pid
            ]
            self.signal_members(
                [member for member in descendants if member.has_live_tasks],
                signal.SIGKILL)
            remaining = deadline - asyncio.get_running_loop().time()
            if descendants and remaining > 0:
                await asyncio.sleep(
                    min(_SUBPROCESS_EXIT_POLL_SECS, remaining))
                continue
            if descendants:
                raise RuntimeError(
                    f'pool {binary_name} subprocess supervisor for job={job_id} '
                    f'did not reap {len(descendants)} descendant(s)')
            # The subreaping leader has no child left to protect. A final KILL
            # is now safe and cannot orphan an owned zombie.
            self.signal_members(
                [member for member in members if member.has_live_tasks],
                signal.SIGKILL)
            return

    async def wait_for_quiescence(self, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            members = self.members()
            self.reap_adopted_descendant_zombies(members)
            members = self.members()
            if not members:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(_SUBPROCESS_EXIT_POLL_SECS, remaining))
        if members:
            raise RuntimeError(
                f'Owned subprocess session sid={self.leader.session_id} still '
                f'has {len(members)} member(s), including zombies')


@dataclasses.dataclass
class _OwnedProcessGroup:
    """Portable degraded ownership when Linux pidfd/procfs is unavailable.

    ``start_new_session=True`` creates the group, so its numeric ID is owned by
    this launch. This fallback cannot enumerate nested process groups or reap
    orphaned grandchildren; production Linux workers therefore use
    ``_OwnedSubprocessSession`` whenever its stronger capabilities are present.
    """
    process_group_id: int
    _termination_attempted: bool = False

    def _exists(self) -> bool:
        try:
            os.killpg(self.process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def _wait_until_absent(self, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while self._exists():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_SUBPROCESS_EXIT_POLL_SECS, remaining))
        return True

    async def terminate(self, binary_name: str, job_id: JobID) -> None:
        if not self._exists():
            return
        self._termination_attempted = True
        try:
            os.killpg(self.process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        if await self._wait_until_absent(_SUBPROCESS_TERM_GRACE_SECS):
            return
        log.warning(
            'pool %s subprocess process group for job=%s did not exit within '
            '%.1fs of SIGTERM; sending SIGKILL.', binary_name, job_id,
            _SUBPROCESS_TERM_GRACE_SECS)
        try:
            os.killpg(self.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        if not await self._wait_until_absent(_SUBPROCESS_KILL_GRACE_SECS):
            # Without procfs/subreaper support, killpg(0) cannot distinguish a
            # live process from an already-killed orphan zombie. The fallback
            # guarantees bounded TERM/KILL delivery, but deliberately does not
            # claim the strong supervisor's exact-reap guarantee.
            log.warning(
                'pool %s subprocess process group for job=%s remained '
                'observable after SIGKILL; degraded fallback cannot prove '
                'whether only unreaped zombies remain.', binary_name, job_id)

    async def wait_for_quiescence(self, timeout: float) -> None:
        if await self._wait_until_absent(timeout):
            return
        if self._termination_attempted:
            return
        raise RuntimeError(
            f'Owned subprocess process group pgid={self.process_group_id} '
            'is still live')


def _log_subprocess_output(record: bytes, binary_name: str,
                           job_id: JobID) -> None:
    if record:
        log.info('[%s/%s] %s', binary_name, job_id,
                 record.decode('utf-8', errors='replace'))


async def _drain_subprocess_output(
    stdout: asyncio.StreamReader,
    binary_name: str,
    job_id: JobID,
) -> None:
    """Drain newline- and carriage-return-delimited subprocess output.

    TQDM rewrites its progress line with ``\r`` and may not emit ``\n`` until
    the job finishes. ``StreamReader.readline`` eventually exceeds its buffer
    limit on that stream, stops consuming the pipe, and blocks the child in
    ``write(2)``. Reading bounded chunks and treating either delimiter as a
    complete record keeps long-running training and prediction processes
    flowing. Undelimited output is also flushed in bounded records so a
    malformed child cannot grow this buffer without limit.
    """
    pending = b''
    progress_record: bytes | None = None
    last_progress_log_at: float | None = None

    def _flush_progress(*, force: bool) -> None:
        nonlocal progress_record, last_progress_log_at
        if progress_record is None:
            return
        now = time.monotonic()
        if (force or last_progress_log_at is None or
                now - last_progress_log_at >=
                _SUBPROCESS_PROGRESS_LOG_INTERVAL_SECS):
            _log_subprocess_output(progress_record, binary_name, job_id)
            progress_record = None
            last_progress_log_at = now

    while True:
        chunk = await stdout.read(_SUBPROCESS_OUTPUT_CHUNK_BYTES)
        if not chunk:
            break
        records = (pending + chunk).splitlines(keepends=True)
        pending = b''
        for record in records:
            if record.endswith(b'\r'):
                # TQDM can emit one carriage-return update per batch. Keep
                # draining all of them, but only publish the latest update at
                # a bounded cadence so a long job does not flood service logs.
                progress_record = record.rstrip(b'\r')
                _flush_progress(force=False)
            elif record.endswith(b'\n'):
                _flush_progress(force=True)
                _log_subprocess_output(record.rstrip(b'\r\n'), binary_name,
                                       job_id)
            else:
                pending = record
        while len(pending) >= _SUBPROCESS_OUTPUT_MAX_RECORD_BYTES:
            _log_subprocess_output(
                pending[:_SUBPROCESS_OUTPUT_MAX_RECORD_BYTES], binary_name,
                job_id)
            pending = pending[_SUBPROCESS_OUTPUT_MAX_RECORD_BYTES:]
        _flush_progress(force=False)
    _flush_progress(force=True)
    _log_subprocess_output(pending, binary_name, job_id)


class _SubprocessSupervisorUnavailable(RuntimeError):
    """The stronger Linux supervisor cannot run on this worker."""


def _subprocess_supervisor_capabilities_available() -> bool:
    """Check capabilities needed before starting the ownership handshake."""
    if not sys.platform.startswith('linux'):
        return False
    pidfd: Optional[int] = None
    try:
        os.stat('/proc/self/ns/pid')
        with open('/proc/self/stat', 'rb') as stat_file:
            stat_file.read(1)
        pidfd = pidfd_open(os.getpid())
        pidfd_send_signal(pidfd, 0)
    except (AttributeError, FileNotFoundError, OSError):
        return False
    finally:
        if pidfd is not None:
            os.close(pidfd)
    return True


async def _launch_process_group_fallback(
    cmd: list[str],
    child_env: dict[str, str],
    binary_name: str,
    job_id: JobID,
) -> tuple[asyncio.subprocess.Process, Optional[_OwnedProcessGroup]]:
    """Start a portable owned group when pidfd/subreaper is unavailable."""
    can_own_group = callable(getattr(os, 'killpg', None))
    log.warning(
        'pool %s subprocess supervisor capabilities are unavailable for '
        'job=%s; using %s.', binary_name, job_id,
        ('bounded process-group TERM/KILL fallback' if can_own_group else
         'bounded direct-child TERM/KILL fallback'))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=child_env,
        start_new_session=can_own_group,
    )
    # Unit tests use a Process stand-in and retain direct-child behavior.
    if (not can_own_group
            or not isinstance(proc, asyncio.subprocess.Process)):
        return proc, None
    return proc, _OwnedProcessGroup(proc.pid)


async def _launch_owned_subprocess(
    cmd: list[str],
    child_env: dict[str, str],
    binary_name: str,
    job_id: JobID,
) -> tuple[
        asyncio.subprocess.Process,
        Optional[_OwnedSubprocessSession | _OwnedProcessGroup],
]:
    """Handshake ownership before the supervisor can start the real CLI."""
    if not _subprocess_supervisor_capabilities_available():
        return await _launch_process_group_fallback(
            cmd, child_env, binary_name, job_id)

    ready_read, ready_write = os.pipe()
    ack_read, ack_write = os.pipe()
    ack_write_fd: Optional[int] = ack_write

    def _close_ack_write() -> None:
        nonlocal ack_write_fd
        if ack_write_fd is None:
            return
        os.close(ack_write_fd)
        ack_write_fd = None

    supervisor_cmd = [
        sys.executable,
        '-m',
        'pool.subprocess_supervisor',
        '--ready-fd',
        str(ready_write),
        '--ack-fd',
        str(ack_read),
        '--term-grace',
        str(_SUBPROCESS_TERM_GRACE_SECS),
        '--kill-grace',
        str(_SUBPROCESS_KILL_GRACE_SECS),
        '--',
        *cmd,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *supervisor_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
            pass_fds=(ready_write, ack_read),
        )
    except BaseException:
        os.close(ready_read)
        _close_ack_write()
        raise
    finally:
        os.close(ready_write)
        os.close(ack_read)

    # Unit tests use a Process stand-in and retain their direct-child behavior.
    if not isinstance(proc, asyncio.subprocess.Process):
        os.close(ready_read)
        _close_ack_write()
        return proc, None

    async def _abort_unacknowledged_supervisor() -> None:
        _close_ack_write()
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                proc.wait(), timeout=_SUBPROCESS_KILL_GRACE_SECS)

    async def _wait_for_unacknowledged_supervisor_abort() -> None:
        cleanup_task = asyncio.create_task(
            _abort_unacknowledged_supervisor(),
            name=f'pool-{binary_name}-handshake-abort')
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        cleanup_task.result()

    async def _read_ready_byte() -> bytes:
        loop = asyncio.get_running_loop()
        ready = loop.create_future()

        def _read() -> None:
            try:
                ready.set_result(os.read(ready_read, 1))
            except BaseException as error:
                ready.set_exception(error)
            finally:
                loop.remove_reader(ready_read)

        loop.add_reader(ready_read, _read)
        try:
            return await ready
        finally:
            loop.remove_reader(ready_read)

    try:
        ready = await asyncio.wait_for(
            _read_ready_byte(),
            timeout=_SUBPROCESS_OWNERSHIP_HANDSHAKE_SECS)
        if ready == b'U':
            raise _SubprocessSupervisorUnavailable(
                'subreaper setup was rejected by the worker runtime')
        if ready != b'R':
            raise RuntimeError(
                f'pool {binary_name} supervisor for job={job_id} exited '
                'before ownership handshake')
        owned_session = _OwnedSubprocessSession.capture(proc.pid)
        leader = owned_session.leader
        log.info(
            'pool %s subprocess ownership for job=%s: '
            'pid=pgid=sid=%s start_time=%s proc=%s:%s pid_ns=%s:%s',
            binary_name, job_id, leader.pid, leader.start_time_ticks,
            leader.proc_device, leader.proc_inode,
            leader.pid_namespace_device, leader.pid_namespace_inode)
        assert ack_write_fd is not None
        os.write(ack_write_fd, b'G')
        return proc, owned_session
    except asyncio.CancelledError:
        await _wait_for_unacknowledged_supervisor_abort()
        raise
    except Exception as error:
        await _wait_for_unacknowledged_supervisor_abort()
        if isinstance(error, _SubprocessSupervisorUnavailable):
            return await _launch_process_group_fallback(
                cmd, child_env, binary_name, job_id)
        raise
    finally:
        os.close(ready_read)
        _close_ack_write()


async def _run_kumo_subprocess(
        binary_name: str, job_id: JobID, job_config_path: str,
        job_output_store: Optional[MLJobOutputStore],
        subprocess_environment: Optional[dict[str, str]] = None) -> JobState:
    """Run ``binary_name`` (kumo-auto / kumo-pred) and wait for it to exit.

    Heartbeats periodically so Temporal sees the activity alive; on
    workflow cancellation, SIGTERM the subprocess (with 10s grace) then
    SIGKILL — without this kumo-auto / kumo-pred would keep running and
    leak GPU + SSD until natural exit.
    """
    # Prefer the repo-local script; fall back to PATH.
    repo_local = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                          'bin', binary_name)
    if osp.exists(repo_local):
        cmd = [sys.executable, repo_local, '--json', job_config_path]
    else:
        cmd = [binary_name, '--json', job_config_path]

    log.info('pool %s subprocess starting for job=%s: %s', binary_name, job_id,
             ' '.join(cmd))
    # PYTHONUNBUFFERED=1: piped stderr is block-buffered by default, which
    # would silence kumo-auto's per-step Python logging until process exit.
    child_env = {
        **os.environ,
        **(subprocess_environment or {}),
        'PYTHONUNBUFFERED': '1',
    }
    if (binary_name == 'kumo-pred'
            and 'KUMO_BP_OUTPUT_SPOOL_ROOT' not in child_env):
        # DiskGraph's configured data root is the authoritative local-storage
        # path inside both single-host workers and activity pods. Keep the
        # bounded Parquet spool on that same local filesystem without exposing
        # the host-side DISKGRAPH_DATA_ROOT bind source to the container.
        data_root = getattr(deploy_env().data_compute, 'data_root', None)
        if data_root is not None:
            child_env['KUMO_BP_OUTPUT_SPOOL_ROOT'] = osp.join(
                data_root, 'bp-output-spool')
    proc, owned_session = await _launch_owned_subprocess(
        cmd, child_env, binary_name, job_id)

    assert proc.stdout is not None
    drain_task = asyncio.create_task(
        _drain_subprocess_output(proc.stdout, binary_name, job_id),
        name=f'pool-{binary_name}-drain')
    proc_wait_task = asyncio.create_task(
        proc.wait(), name=f'pool-{binary_name}-wait')

    async def _terminate_subprocess() -> None:
        if owned_session is not None:
            await owned_session.terminate(binary_name, job_id)
        elif proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.shield(proc_wait_task),
                    timeout=_SUBPROCESS_TERM_GRACE_SECS)
            except asyncio.TimeoutError:
                log.warning(
                    'pool %s subprocess for job=%s did not exit within %.1fs '
                    'of SIGTERM; sending SIGKILL.', binary_name, job_id,
                    _SUBPROCESS_TERM_GRACE_SECS)
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

        # asyncio's Process.wait() can remain pending after the direct child
        # exits while a descendant still owns the stdout pipe. Session cleanup
        # above closes those inherited descriptors before this bounded reap.
        if not proc_wait_task.done():
            await asyncio.wait_for(
                asyncio.shield(proc_wait_task),
                timeout=_SUBPROCESS_KILL_GRACE_SECS)

    async def _cancel_output_drain() -> None:
        if not drain_task.done():
            drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            # Expected after cancelling an active drain task above.
            pass
        except Exception:
            # Supervision is already failing or being cancelled. Preserve that
            # original exception while retaining visibility into drain cleanup.
            log.warning(
                'pool %s output drain cleanup failed for job=%s', binary_name,
                job_id, exc_info=True)

    async def _cleanup_subprocess() -> None:
        await _terminate_subprocess()
        await _cancel_output_drain()

    async def _run_cleanup_uncancelled() -> None:
        """Finish ownership cleanup even if cancellation is delivered again."""
        cleanup_task = asyncio.create_task(
            _cleanup_subprocess(), name=f'pool-{binary_name}-cleanup')
        while True:
            try:
                await asyncio.shield(cleanup_task)
                return
            except asyncio.CancelledError:
                if cleanup_task.done():
                    # Preserve cancellation raised by cleanup itself while
                    # swallowing only repeated cancellation of this activity.
                    cleanup_task.result()
                    return
                log.warning(
                    'pool %s cleanup for job=%s was cancelled again; waiting '
                    'for bounded process-session cleanup.', binary_name, job_id)

    cancellation_requested = False
    try:
        while not proc_wait_task.done():
            watched = {proc_wait_task}
            if not drain_task.done():
                watched.add(drain_task)
            done, _ = await asyncio.wait(
                watched,
                timeout=_SUBPROCESS_HEARTBEAT_INTERVAL_SECS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if drain_task in done:
                # Propagate a broken output drain immediately. Otherwise the
                # child can fill its stdout pipe and block indefinitely.
                await drain_task
            if proc_wait_task in done:
                break
            if not done:
                activity.heartbeat()
                if binary_name == 'kumo-auto':
                    # train always supplies the store; only the kumo-auto
                    # branch publishes, so this is never None here.
                    assert job_output_store is not None
                    try:
                        await asyncio.wait_for(
                            publish_training_progress(job_id, job_config_path,
                                                      job_output_store),
                            timeout=TRAINING_PROGRESS_PUBLISH_TIMEOUT_SECS,
                        )
                    except Exception:
                        log.exception(
                            'Failed to publish pool training progress for '
                            'job=%s', job_id)
        await proc_wait_task
    except asyncio.CancelledError:
        cancellation_requested = True
        log.info(
            'pool %s subprocess for job=%s: workflow cancelled, '
            'terminating subprocess (pid=%s).', binary_name, job_id, proc.pid)
        try:
            await _run_cleanup_uncancelled()
        except BaseException:
            # Cancellation is the public activity outcome. Cleanup failures are
            # still visible, and finalization below makes one last bounded
            # attempt, but they must not replace CancelledError.
            log.exception(
                'pool %s cancellation cleanup failed for job=%s', binary_name,
                job_id)
        raise
    except Exception:
        log.exception(
            'pool %s subprocess supervision failed for job=%s; '
            'terminating subprocess.', binary_name, job_id)
        await _cleanup_subprocess()
        raise
    else:
        # The process has exited; consume all output before reporting success.
        await drain_task
    finally:
        if owned_session is not None:
            async def _finalize_owned_session() -> None:
                if isinstance(owned_session, _OwnedProcessGroup):
                    # The fallback directly runs the CLI, so its parent may
                    # exit before a descendant. Reclaim the group proactively.
                    if not owned_session._termination_attempted:
                        await owned_session.terminate(binary_name, job_id)
                    await owned_session.wait_for_quiescence(
                        _SUBPROCESS_KILL_GRACE_SECS)
                    return
                try:
                    await owned_session.wait_for_quiescence(
                        _SUBPROCESS_KILL_GRACE_SECS)
                except Exception:
                    # Preserve the quiescence error, but first reclaim an
                    # exact session whose supervisor exited unexpectedly.
                    await owned_session.terminate(binary_name, job_id)
                    await owned_session.wait_for_quiescence(
                        _SUBPROCESS_KILL_GRACE_SECS)
                    raise

            quiescence_task = asyncio.create_task(
                _finalize_owned_session(),
                name=f'pool-{binary_name}-quiescence')
            while not quiescence_task.done():
                try:
                    await asyncio.shield(quiescence_task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    # Inspect the completed task below so cancellation can
                    # remain the activity outcome while preserving diagnostics.
                    break
            try:
                quiescence_task.result()
            except (Exception, asyncio.CancelledError):
                if not cancellation_requested:
                    raise
                log.exception(
                    'pool %s final cancellation quiescence failed for job=%s',
                    binary_name, job_id)
        if proc.returncode is None:
            # A second cancellation can interrupt the graceful termination
            # await above. Always deliver a final synchronous SIGKILL before
            # abandoning supervision so the GPU/SSD-holding child cannot leak.
            log.warning(
                'pool %s subprocess for job=%s still alive during cleanup; '
                'sending SIGKILL.', binary_name, job_id)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        if not proc_wait_task.done():
            proc_wait_task.cancel()
        try:
            await proc_wait_task
        except asyncio.CancelledError:
            # Expected when final cleanup cancels a still-pending wait task.
            pass
    rc = proc.returncode
    log.info('pool %s subprocess for job=%s exited with rc=%s', binary_name,
             job_id, rc)
    return JobState.SUCCEEDED if rc == 0 else JobState.FAILED


_PLACEMENT_ERROR_MESSAGES = {
    'ProvisionTimeoutError':
    ('Your job could not acquire a GPU/compute worker within the '
     'allowed time. GPU capacity may be temporarily limited. '
     'Please retry; if the problem persists, contact Kumo support.'),
    'ProvisionTerminalError':
    ('Your job could not start because the compute pod failed to '
     'initialize (e.g. bad image tag, missing secret, or invalid '
     'container configuration). This is an infrastructure issue — '
     'please contact Kumo support.'),
    'InsufficientDataplaneCapacityError':
    ('Your graph cannot be ingested because the local dataplane volume '
     'does not have enough available space. Free or expand that volume, '
     'move the dataplane root to supported object storage, or reduce the '
     'dataset size.'),
    'ExistingWorkerUnavailableError':
    ('The DiskGraph worker reserved for your job is no longer available. '
     'This is an infrastructure issue, not a problem with your graph or '
     'model. Please retry; if the problem persists, contact Kumo support.'),
}


def _raise_as_application_error(ex: _RpcError) -> None:
    """Re-raise an AutoRPC ``RpcError`` as ``ApplicationError`` so that
    Temporal's activity-retry policy honors the ``non_retryable`` bit.

    Without this translation, ``RpcError`` propagates as a generic
    exception whose type-name is not in the default
    ``non_retryable_error_types`` list, and Temporal retries
    deterministic controller failures (RBAC denial, missing IRSA,
    invalid spec) for the full ``schedule_to_close`` budget.
    """
    raise ApplicationError(ex.message, type=ex.error_type,
                           non_retryable=ex.non_retryable) from ex


@activity.defn
async def place_snapshot_activity(request: PlacementRequest) -> PlacementResult:
    from statedb.config.common import DataSnapshot
    request = dataclasses.replace(
        request,
        snapshot_id=DataSnapshot.GRAPH.canonicalize(request.snapshot_id),
    )
    client = get_controller_client(deploy_env())
    try:
        return await client.place_snapshot(request)
    except _RpcError as ex:
        # Pool-placement errors carry a user-facing message so that when
        # place_snapshot_activity exhausts its retry budget and Temporal
        # surfaces a ScheduleToClose TimeoutError, _timeout_to_public_exception
        # finds a PublicException in __cause__ and shows the specific reason
        # (GPU capacity unavailable / bad pod config) instead of the generic
        # "data too large" fallback.
        public_msg = _PLACEMENT_ERROR_MESSAGES.get(ex.error_type)
        if public_msg is not None:
            raise ApplicationError(
                ex.message,
                PublicException(message=public_msg).to_dict(),
                type=ex.error_type,
                non_retryable=ex.non_retryable,
            ) from ex
        _raise_as_application_error(ex)
        raise  # unreachable; keeps the static checker happy


@activity.defn
async def release_snapshot_activity(snapshot_id: SnapshotID,
                                    job_id: JobID) -> None:
    from statedb.config.common import DataSnapshot
    snapshot_id = DataSnapshot.GRAPH.canonicalize(snapshot_id)
    client = get_controller_client(deploy_env())
    try:
        await client.release_lease(snapshot_id, job_id)
    except _RpcError as ex:
        _raise_as_application_error(ex)


@activity.defn
async def get_pool_stats_activity(
        filter: Optional[PoolStatsFilter] = None) -> PoolStats:
    """Read-only pool-stats probe used by ``run_pooled_activity`` to poll a
    placed worker's health after an activity failure (STALE-wait / READY-retry
    / DEAD-reroute). Thin wrapper around the controller RPC so workflow code
    never imports the controller client directly."""
    client = get_controller_client(deploy_env())
    try:
        return await client.get_pool_stats(filter)
    except _RpcError as ex:
        _raise_as_application_error(ex)
        raise  # unreachable; keeps the static checker happy
