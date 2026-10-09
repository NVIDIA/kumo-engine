# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kumo <-> DiskGraph runtime boundary — :class:`DiskgraphService`.

Wraps the in-process DiskGraph pybind runtime (single concrete class; pybind
runs in every DG pod). :class:`DiskgraphHandle` is snapshot identity
``(graph_key, graph_data_version)``, which the service resolves to an on-VM
path under ``data_root``.

Heavy sub-module deps are imported inside method bodies (not at module scope)
to avoid a circular-import cascade through ``kumo.diskgraph.__init__``.
"""

import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse

from diskgraph import TimeUnit
from kumo_core.task_dataset import (
    with_resolved_target_encoder_input,
    write_flat_manifest,
    write_thin_manifest,
)

from async_utils.sync import blocking_method
from io_utils import kumo_dataclass
from io_utils.json import from_json_dict, to_json_dict
# DiskgraphHandle lives in kumo.diskgraph._backend (so GlobalEngineState can type
# it); re-exported here as the canonical import path.
from kumo.diskgraph._backend import DiskgraphHandle
from service_framework._raw_deploy_config import DiskgraphConfig

logger = logging.getLogger(__name__)

_DOWNLOAD_INVENTORY_SCHEMA = 'kumo.diskgraph_download_inventory.v1'
_DOWNLOAD_SUCCESS_FILENAME = '_DOWNLOAD_SUCCESS'
_MAX_DOWNLOAD_MARKER_BYTES = 32 * 1024 * 1024
_MAX_DOWNLOAD_FILES = 100_000
_DOWNLOAD_BACKUP_SUFFIX = '.previous'
_EMPTY_SOURCE_CONTROL_FILENAMES = frozenset({'_SUCCESS'})
_EMPTY_SOURCE_CONTROL_PREFIXES = ('_committed_', '_started_')

_RESTORE_SUCCESS_FILENAME = '_SUCCESS'
# The activity's heartbeat_timeout is 300s. `fsutil.copy` reports no
# progress, and one object can occupy it for minutes -- copy_file's retry
# wrapping the provider's allows six attempts -- so a ticker is the only
# signal available.
_RESTORE_HEARTBEAT_INTERVAL_SECS = 30.0


def _start_heartbeat_ticker(
        heartbeat: Callable[[], None]) -> Tuple[threading.Thread, threading.Event]:
    stop = threading.Event()

    def _tick() -> None:
        while not stop.wait(_RESTORE_HEARTBEAT_INTERVAL_SECS):
            heartbeat()

    ticker = threading.Thread(target=_tick, name='diskgraph-restore-heartbeat',
                              daemon=True)
    ticker.start()
    return ticker, stop


def _copy_dataplane_mirror(
    dataplane_dir: str,
    output_dir: str,
    *,
    heartbeat: Optional[Callable[[], None]] = None,
) -> None:
    """Copy the dataplane mirror to local SSD.

    ``_SUCCESS`` is excluded from the bulk copy and copied on its own
    afterwards, so the local copy preserves the producer's invariant: if
    ``_SUCCESS`` is present, everything else already is.
    """
    from io_utils import fsutil

    # A stale ``_SUCCESS`` from an earlier attempt must not outlive the copy
    # it precedes, or it marks a half-restored tree ready for the duration.
    with suppress(FileNotFoundError):
        os.remove(os.path.join(output_dir, _RESTORE_SUCCESS_FILENAME))

    # The activity has already spent its SSD-residency check by now.
    if heartbeat is not None:
        heartbeat()

    ticker: Optional[threading.Thread] = None
    stop_ticker: Optional[threading.Event] = None
    if heartbeat is not None:
        ticker, stop_ticker = _start_heartbeat_ticker(heartbeat)

    start = time.monotonic()
    try:
        fsutil.copy(dataplane_dir, output_dir, overwrite=True,
                    raise_on_exists=False, exclude=[_RESTORE_SUCCESS_FILENAME])
        fsutil.copy_file(
            f'{dataplane_dir}/{_RESTORE_SUCCESS_FILENAME}',
            os.path.join(output_dir, _RESTORE_SUCCESS_FILENAME),
            overwrite=True, raise_on_exists=False)
    finally:
        if stop_ticker is not None:
            stop_ticker.set()
        if ticker is not None:
            ticker.join()

    elapsed = max(time.monotonic() - start, 1e-9)
    # Measured from what actually landed on disk -- there is no manifest to
    # read sizes from anymore.
    num_files = 0
    total_bytes = 0
    for root, _, filenames in os.walk(output_dir):
        for filename in filenames:
            total_bytes += os.path.getsize(os.path.join(root, filename))
            num_files += 1
    logger.info(
        'restore_snapshot: copied %d files / %d bytes into %s took %.1fs '
        '(%.1f MB/s)', num_files, total_bytes, output_dir, elapsed,
        total_bytes / elapsed / 1e6)


def _download_inventory_context(alias: str, table: Any,
                                data_version: Any) -> dict[str, str]:
    return {
        'alias': str(alias),
        'connector_id': str(table.get_connector_id()),
        'source_table': str(table.get_source_table_name()),
        'table_fingerprint': str(table.fingerprint()),
        'data_version': str(data_version),
    }


def _local_download_inventory(local_dest: str) -> list[dict[str, Any]]:
    """Capture the exact local source set DG can enumerate on retry."""
    inventory = []
    if os.path.islink(local_dest):
        raise RuntimeError(
            f'DiskGraph staged source root is a symlink: {local_dest!r}.')
    root = os.path.realpath(local_dest)
    for directory, subdirs, filenames in os.walk(local_dest, followlinks=False):
        for subdir in subdirs:
            path = os.path.join(directory, subdir)
            if os.path.islink(path):
                raise RuntimeError(
                    f'DiskGraph staged source contains a symlink directory: '
                    f'{path!r}.')
            resolved = os.path.realpath(path)
            if os.path.commonpath((root, resolved)) != root:
                raise RuntimeError(
                    f'DiskGraph staged source directory escapes its root: '
                    f'{path!r}.')
        subdirs.sort()
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            if path == os.path.join(local_dest, _DOWNLOAD_SUCCESS_FILENAME):
                continue
            if os.path.islink(path):
                raise RuntimeError(
                    f'DiskGraph staged source contains a symlink: {path!r}.')
            relative_path = os.path.relpath(path, local_dest)
            resolved = os.path.realpath(path)
            if os.path.commonpath((root, resolved)) != root:
                raise RuntimeError(
                    f'DiskGraph staged source escapes its root: {path!r}.')
            file_stat = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(
                    f'DiskGraph staged source is not a regular file: {path!r}.')
            is_control_marker = (
                filename in _EMPTY_SOURCE_CONTROL_FILENAMES
                or filename.startswith(_EMPTY_SOURCE_CONTROL_PREFIXES))
            if file_stat.st_size <= 0 and not is_control_marker:
                raise RuntimeError(
                    f'DiskGraph staged source contains an empty file: {path!r}.')
            inventory.append({
                'path': relative_path,
                'size_bytes': file_stat.st_size,
                'device': file_stat.st_dev,
                'inode': file_stat.st_ino,
                'mtime_ns': file_stat.st_mtime_ns,
            })
            if len(inventory) > _MAX_DOWNLOAD_FILES:
                raise RuntimeError(
                    f'DiskGraph staged source exceeds the {_MAX_DOWNLOAD_FILES} '
                    'file inventory limit.')
    if not inventory:
        raise RuntimeError(
            f'DiskGraph staged source at {local_dest!r} contains no files.')
    return inventory


def _download_inventory_payload(alias: str, table: Any, data_version: Any,
                                local_dest: str) -> dict[str, Any]:
    return {
        'schema_version': _DOWNLOAD_INVENTORY_SCHEMA,
        'context': _download_inventory_context(alias, table, data_version),
        'files': _local_download_inventory(local_dest),
    }


def _validate_download_inventory(marker: str, alias: str, table: Any,
                                 data_version: Any,
                                 local_dest: str) -> tuple[bool, str]:
    try:
        with open(marker, 'rb') as source:
            raw = source.read(_MAX_DOWNLOAD_MARKER_BYTES + 1)
        if len(raw) > _MAX_DOWNLOAD_MARKER_BYTES:
            return False, 'marker exceeds the bounded byte limit'
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return False, 'marker must be a JSON object'
        if payload.get('schema_version') != _DOWNLOAD_INVENTORY_SCHEMA:
            return False, 'missing or unsupported marker schema'
        if payload.get('context') != _download_inventory_context(
                alias, table, data_version):
            return False, 'marker belongs to a different table attempt'
        declared = payload.get('files')
        if not isinstance(declared, list):
            return False, 'marker has no file inventory'
        if len(declared) > _MAX_DOWNLOAD_FILES:
            return False, 'marker exceeds the bounded file-count limit'
        observed = _local_download_inventory(local_dest)
        if declared != observed:
            return False, 'staged file inventory changed after publication'
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        return False, f'marker validation failed: {error}'
    return True, ''


def _write_download_inventory_marker(marker: str, payload: dict[str,
                                                                Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode()
    if len(encoded) > _MAX_DOWNLOAD_MARKER_BYTES:
        raise RuntimeError(
            'DiskGraph staged source inventory exceeds the bounded marker '
            'byte limit.')
    fd, temporary = tempfile.mkstemp(
        prefix=f'.{_DOWNLOAD_SUCCESS_FILENAME}.',
        suffix='.tmp',
        dir=os.path.dirname(marker),
    )
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, marker)
        directory_fd = os.open(os.path.dirname(marker), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _remove_owned_download_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.unlink(path)


def _remove_stale_download_attempts(local_dest: str) -> None:
    """Remove incomplete connector attempts while snapshot ownership is held."""
    normalized_dest = os.path.normpath(local_dest)
    parent = os.path.dirname(normalized_dest)
    prefix = f'.{os.path.basename(normalized_dest)}.download-'
    if not os.path.isdir(parent):
        return
    removed = False
    with os.scandir(parent) as entries:
        for entry in entries:
            if entry.name.startswith(prefix):
                _remove_owned_download_path(entry.path)
                removed = True
    if removed:
        _fsync_parent(normalized_dest)


def _fsync_parent(path: str) -> None:
    directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _recover_interrupted_download_swap(alias: str, table: Any,
                                       data_version: Any,
                                       local_dest: str) -> None:
    """Resolve the deterministic backup left by a process-level crash."""
    backup = f'{local_dest}{_DOWNLOAD_BACKUP_SUFFIX}'
    if not os.path.lexists(backup):
        return

    if os.path.lexists(local_dest):
        valid, _reason = _validate_download_inventory(
            os.path.join(local_dest, _DOWNLOAD_SUCCESS_FILENAME),
            alias,
            table,
            data_version,
            local_dest,
        )
        if valid:
            _remove_owned_download_path(backup)
            _fsync_parent(local_dest)
            return
        _remove_owned_download_path(local_dest)

    os.replace(backup, local_dest)
    _fsync_parent(local_dest)


def _publish_download_attempt(alias: str, table: Any, data_version: Any,
                              attempt_dir: str, local_dest: str) -> None:
    """Swap a complete attempt into place with exception-safe rollback."""
    backup = f'{local_dest}{_DOWNLOAD_BACKUP_SUFFIX}'
    if os.path.lexists(backup):
        raise RuntimeError(
            f'DiskGraph staged source backup was not recovered: {backup!r}.')
    had_previous = os.path.lexists(local_dest)
    if had_previous:
        os.replace(local_dest, backup)
        _fsync_parent(local_dest)

    try:
        os.replace(attempt_dir, local_dest)
        _fsync_parent(local_dest)
    except BaseException:
        # `replace` may have completed before an acknowledgement/error. Accept
        # only an exact newly-published inventory; otherwise restore the old
        # owned cache before propagating the failure.
        published = False
        if os.path.lexists(local_dest):
            published, _reason = _validate_download_inventory(
                os.path.join(local_dest, _DOWNLOAD_SUCCESS_FILENAME),
                alias,
                table,
                data_version,
                local_dest,
            )
        if not published:
            _remove_owned_download_path(local_dest)
            if had_previous and os.path.lexists(backup):
                os.replace(backup, local_dest)
                _fsync_parent(local_dest)
            raise

    if os.path.lexists(backup):
        _remove_owned_download_path(backup)
        _fsync_parent(local_dest)


def _to_diskgraph_dataplane_path(path: str) -> str:
    """Convert Kumo ADLS URIs to DiskGraph's Azure object-store URI form."""
    parsed = urlparse(path)
    if parsed.scheme not in ('abfs', 'abfss'):
        return path
    if '@' not in parsed.netloc:
        return path
    container, account_host = parsed.netloc.split('@', 1)
    account = account_host.split('.', 1)[0]
    key = parsed.path.lstrip('/')
    if not container or not account:
        return path
    return f"az://{account}/{container}/{key}" if key else (
        f"az://{account}/{container}")


def _normalize_direct_object_source(source: dict[str, Any]) -> dict[str, Any]:
    """Translate provider coordinates at the Kumo/DiskGraph boundary.

    Kumo's public ADLS coordinate is ``abfss://container@account...`` while
    DiskGraph's object-store adapter intentionally uses
    ``az://account/container/...``. Keep discovery and identity capture on the
    customer-facing URI, then persist the equivalent DiskGraph coordinate in
    the exact source manifest used by retries.
    """
    if source.get('type') != 'object_store_parquet':
        return source
    objects = source.get('objects')
    if not isinstance(objects, list):
        return source
    return {
        **source,
        'objects': [{
            **obj,
            'path': _to_diskgraph_dataplane_path(obj['path']),
        } for obj in objects],
    }


def _prepare_direct_object_source(
    source: dict[str, Any],
    alias: str,
) -> tuple[dict[str, Any], int]:
    """Normalize a manifest and total its bytes for Kumo admission checks.

    DiskGraph owns source-schema and immutable-identity validation.
    """
    try:
        source = _normalize_direct_object_source(source)
        source_size = sum(int(obj['size']) for obj in source['objects'])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f'Direct object manifest for graph table {alias!r} is malformed.'
        ) from error
    if source_size <= 0:
        raise RuntimeError(
            f'Direct object manifest for graph table {alias!r} is empty.')
    return source, source_size


def _normalized_local_path(path: str) -> str:
    """Normalize a DiskGraph-local path before comparing manifest bindings."""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _close_connector_quietly(connector: Any) -> None:
    """Release a resolved connector without replacing source failures."""
    with suppress(Exception):
        connector.close()


def _validate_ttng_ingest_binding(
    ingest_manifest: dict[str, Any],
    task_dataset: dict[str, Any],
    *,
    graph_index_path: str,
    tt_extension_path: str,
    tt_id: str,
) -> None:
    """Reject an ingest manifest that is not bound to the expected snapshot.

    ``execute_tt_ingest`` writes all of these values into its published
    manifest.  Checking both the top-level result and its task-dataset block
    prevents a stale extension, an extension copied from another snapshot, or
    a mismatched per-stage table from being silently materialized.

    NOTE: this is a path/tt_id *sanity* gate, not the cross-graph correctness
    guard. The stamped ``graph_index_path`` is written at ingest time with the
    same index the re-ingest targets, so it cannot detect "seeds generated
    against graph A, trained against graph B". That correctness comes from
    ingest-time FK membership verification: ``execute_tt_ingest`` verifies
    every seed FK against the training graph's node data, skips + counts
    orphans (``key_failures`` in the manifest/result), and
    ``ensure_stage_ingested`` warns / optionally fails on the skipped
    fraction. Do not re-introduce a "graphs match" assumption on top of this
    check.
    """
    expected_graph_index = _normalized_local_path(graph_index_path)
    expected_extension = _normalized_local_path(tt_extension_path)

    def _require_path(container: dict[str, Any], field: str,
                      expected: str) -> None:
        value = container.get(field)
        if not isinstance(value,
                          str) or _normalized_local_path(value) != expected:
            raise RuntimeError(
                'TT-NG ingest manifest binding mismatch for '
                f'{field!r}; expected {expected!r}, got {value!r}. '
                'Re-ingest this task dataset for the current graph snapshot.')

    if ingest_manifest.get('tt_id') != tt_id:
        raise RuntimeError(
            'TT-NG ingest manifest binding mismatch for \'tt_id\'; expected '
            f'{tt_id!r}, got {ingest_manifest.get("tt_id")!r}. Re-ingest this '
            'task dataset for the current graph snapshot.')
    _require_path(ingest_manifest, 'graph_index_path', expected_graph_index)
    _require_path(ingest_manifest, 'tt_extension_path', expected_extension)
    _require_path(task_dataset, 'graph_index_path', expected_graph_index)
    _require_path(task_dataset, 'tt_extension_path', expected_extension)

    diskgraph_manifest_path = ingest_manifest.get('diskgraph_manifest_path')
    expected_manifest = _normalized_local_path(
        os.path.join(tt_extension_path, 'diskgraph_manifest.json'))
    if (not isinstance(diskgraph_manifest_path, str)
            or _normalized_local_path(diskgraph_manifest_path)
            != expected_manifest):
        raise RuntimeError(
            'TT-NG ingest manifest binding mismatch for '
            "'diskgraph_manifest_path'; re-ingest this task dataset for the "
            'current graph snapshot.')

    stages = task_dataset.get('stages')
    if not isinstance(stages, dict) or not stages:
        raise RuntimeError(
            'TT-NG ingest manifest is missing task_dataset.stages; re-ingest '
            'this task dataset for the current graph snapshot.')
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            raise RuntimeError(
                'TT-NG ingest manifest has malformed task_dataset stage '
                f'{stage_name!r}; re-ingest this task dataset.')
        _require_path(stage, 'index_path', expected_extension)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@kumo_dataclass
class IngestResult:
    """Output of :meth:`DiskgraphService.ingest`.

    The four fields are filesystem paths under the configured
    ``data_root``.
    """
    index_path: str
    col_stats_path: str
    ingest_stats_path: str
    kumo_stats_dir: str
    pkey_duplicate_counts: Optional[dict[str, int]] = None


@kumo_dataclass
class TrainTableResult:
    output_path: str
    manifest_path: str


@kumo_dataclass
class PredTableResult:
    total_rows: int
    total_partitions: int


@kumo_dataclass
class MaterializeResult:
    """Output of :meth:`DiskgraphService.materialize_task_dataset`.

    ``manifest_path`` is the sidecar index; ``runtime_manifest_path`` is the
    flat manifest the wheel iterator opens.
    """
    manifest_path: str
    runtime_manifest_path: str
    bundle_dir: str
    storage_kind: str
    # BP path: predict-stage row count from ingest (replaces pred_table read).
    predict_row_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DiskgraphService:
    """Operations on a DiskGraph snapshot via the in-process pybind runtime.

    Dispatches each operation to the appropriate sub-module:

    * :mod:`kumo.diskgraph.ingest` — graph ingest entry point
    * :mod:`kumo_core.diskgraph.snapshot` — ``_SUCCESS`` read/write
    * :mod:`kumo.data.generic.diskgraph_train_table` /
      :mod:`kumo.data.generic.diskgraph_batch_pred` — training- and
      prediction-table generation
    """
    def __init__(self, config: DiskgraphConfig) -> None:
        self._config = config

    def local_snapshot_path(self, handle: DiskgraphHandle) -> str:
        """Resolve a handle to its on-VM filesystem path."""
        return os.path.join(
            self._config.data_root,
            handle.graph_key,
            handle.graph_data_version,
        )

    @blocking_method()
    def list_ready_handles(self) -> 'list[DiskgraphHandle]':
        """Return handles for every snapshot dir with a valid ``_SUCCESS``
        marker. Inverse of :meth:`local_snapshot_path`; callers should cache the
        result rather than rescanning.
        """
        from diskgraph import DiskgraphSystemError
        from kumo_core.diskgraph.snapshot import check_diskgraph_output
        data_root = self._config.data_root
        ready: 'list[DiskgraphHandle]' = []
        try:
            graph_keys = os.listdir(data_root)
        except OSError:
            return ready
        for graph_key in graph_keys:
            gk_dir = os.path.join(data_root, graph_key)
            if not os.path.isdir(gk_dir):
                continue
            try:
                versions = os.listdir(gk_dir)
            except OSError:
                continue
            for version in versions:
                snap_dir = os.path.join(gk_dir, version)
                if not os.path.isdir(snap_dir):
                    continue
                try:
                    result = check_diskgraph_output(snap_dir)
                except (OSError, DiskgraphSystemError):
                    logger.warning(
                        'Skipping inaccessible DiskGraph snapshot at %s',
                        snap_dir,
                        exc_info=True,
                    )
                    continue
                if result is None:
                    continue
                ready.append(
                    DiskgraphHandle(graph_key=graph_key,
                                    graph_data_version=version))
        return ready

    @staticmethod
    def dataplane_snapshot_path(handle: DiskgraphHandle,
                                dataplane_root: str) -> str:
        """Resolve a handle to its dataplane mirror path
        ``<dataplane_root>/diskgraph_snapshots/<graph_key>/<graph_data_version>/``.
        Owning the convention here keeps Rust dumb about layout.
        """
        return os.path.join(
            dataplane_root,
            'diskgraph_snapshots',
            handle.graph_key,
            handle.graph_data_version,
        )

    @staticmethod
    def diskgraph_dataplane_snapshot_path(handle: DiskgraphHandle,
                                          dataplane_root: str) -> str:
        """Resolve the dataplane mirror path in DiskGraph object-store form."""
        return _to_diskgraph_dataplane_path(
            DiskgraphService.dataplane_snapshot_path(handle, dataplane_root))

    # Internal aliases so existing callers in this module keep working.
    _resolve = local_snapshot_path
    _dataplane_path = dataplane_snapshot_path

    # ------------------------------------------------------------------
    # Ingest + lifecycle
    # ------------------------------------------------------------------

    def ingest(
        self,
        graph_config: Any,
        handle: DiskgraphHandle,
        *,
        data_persist_dir: str,
        table_data_versions: dict,
        dataplane_root: str,
        source_paths: Optional[dict] = None,
        source_specs: Optional[dict] = None,
        subprocess_environment: Optional[dict[str, Optional[str]]] = None,
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> IngestResult:
        """Ingest source tables into a DiskGraph snapshot.

        Rust stream-mirrors each output to the dataplane and writes ``_SUCCESS``
        (the JSON ``IngestResult``) atomically last on both backends.
        """
        from kumo_core.diskgraph.snapshot import DiskgraphIngestResult

        from kumo.diskgraph.ingest import run_diskgraph_ingest

        output_dir = self._resolve(handle)
        # Predict the IngestResult so Rust writes it into ``_SUCCESS``.
        predicted = DiskgraphIngestResult(
            index_path=output_dir,
            col_stats_path=os.path.join(output_dir, 'col_stats.json'),
            ingest_stats_path=os.path.join(output_dir, 'ingest_stats.json'),
            kumo_stats_dir=os.path.join(output_dir, 'kumo_col_stats'),
        )
        output_dataplane_dir = self.diskgraph_dataplane_snapshot_path(
            handle, dataplane_root)
        success_marker_content = json.dumps(to_json_dict(predicted))
        dg = run_diskgraph_ingest(
            graph_config=graph_config,
            data_persist_dir=data_persist_dir,
            table_data_versions=table_data_versions,
            output_dir=output_dir,
            source_paths=source_paths,
            source_specs=source_specs,
            subprocess_environment=subprocess_environment,
            heartbeat=heartbeat,
            output_dataplane_dir=output_dataplane_dir,
            success_marker_content=success_marker_content,
        )
        return IngestResult(
            index_path=dg.index_path,
            col_stats_path=dg.col_stats_path,
            ingest_stats_path=dg.ingest_stats_path,
            kumo_stats_dir=dg.kumo_stats_dir,
        )

    @blocking_method()
    def prepare_direct_sources(
        self,
        connector_specs: Any,
        all_graph_tables: Any,
        *,
        data_persist_dir: str,
        table_data_versions: dict[str, str],
        source_size_bytes_by_table: dict[str, int],
    ) -> dict[str, dict]:
        """Load the frozen direct object sources for this graph."""
        from kumo.connector.file_connector import FileConnector
        from kumo.data.backend_cache import (
            CACHE_DIR,
            DIRECT_DISKGRAPH_SOURCE_DECISION,
            DirectDiskgraphSourceDecision,
        )
        from kumo.data.cache import ReadThroughTableCache
        from rest.config.table_v2 import is_file_source_table
        from rest.utils.driver_common import get_connector_from_spec

        connector_by_id: dict[str, Any] = {}
        source_specs: dict[str, dict] = {}
        source_decisions = ReadThroughTableCache[DirectDiskgraphSourceDecision](
            os.path.join(data_persist_dir, CACHE_DIR),
            name=DIRECT_DISKGRAPH_SOURCE_DECISION,
        )
        with ExitStack() as connector_stack:
            for alias, table in all_graph_tables.items():
                connector_id = str(table.get_connector_id())
                if table.is_sql:
                    continue
                if is_file_source_table(table):
                    data_version = str(table_data_versions[alias])
                    decision = source_decisions.get(
                        table.fingerprint(),
                        data_version,
                        cls=DirectDiskgraphSourceDecision,
                    )
                    if decision is not None:
                        source = decision.source
                        if source is None:
                            continue
                        source, source_size = (
                            _prepare_direct_object_source(source, alias))
                        expected_size = source_size_bytes_by_table.get(alias)
                        if expected_size is None or int(
                                expected_size) != source_size:
                            raise RuntimeError(
                                f'Frozen direct source size for graph '
                                f'table {alias!r} is {source_size}, but '
                                f'table refresh recorded {expected_size!r}.'
                            )
                        source_specs[alias] = {
                            'source': source,
                            'source_size': source_size,
                        }
                        continue

                    # Compatibility for table refreshes completed before
                    # source decisions were persisted. New refreshes always
                    # take the branch above.
                if connector_id not in connector_by_id:
                    reference = connector_specs[connector_id]
                    connector = get_connector_from_spec(reference)
                    connector_stack.callback(_close_connector_quietly,
                                             connector)
                    connector_by_id[connector_id] = connector
                connector = connector_by_id[connector_id]
                if not isinstance(connector, FileConnector):
                    continue
                source_name = table.get_source_table_name()
                if not connector.uses_direct_diskgraph_source_for(
                        source_name):
                    continue
                source = connector.diskgraph_source(source_name=source_name)
                source, source_size = _prepare_direct_object_source(
                    source, alias)

                source_specs[alias] = {
                    'source': source,
                    'source_size': source_size,
                }
        return source_specs

    @blocking_method()
    def prepare_subprocess_environment(
        self,
        connector_specs: Any,
        all_graph_tables: Any,
        *,
        source_specs: Optional[dict[str, Any]],
        data_persist_dir: str,
        table_data_versions: dict[str, str],
    ) -> dict[str, Optional[str]]:
        """Prepare the child-only environment for object-store access."""
        from io_utils.fsutil.s3_connection import (
            AWS_WORKLOAD_CREDENTIAL_ENVIRONMENT_KEYS,
            AWS_WORKLOAD_CREDENTIAL_PROFILE_ISOLATION_ENVIRONMENT,
            s3_subprocess_environment,
            scrubbed_s3_environment,
        )
        from kumo.connector.file_connector import FileConnector
        from kumo.data.backend_cache import (
            CACHE_DIR,
            DIRECT_DISKGRAPH_SOURCE_DECISION,
            DirectDiskgraphSourceDecision,
        )
        from kumo.data.cache import ReadThroughTableCache
        from rest.config.table_v2 import is_file_source_table
        from rest.utils.driver_common import get_connector_from_spec
        from service_framework.deploy_env_config import deploy_env

        direct_source_aliases: set[str]
        compatibility_aliases: set[str] = set()
        if source_specs is not None:
            # Fresh ingest: use the exact aliases handed to the converter.
            direct_source_aliases = set(source_specs)
        else:
            # Retry: recover the frozen refresh-time decision rather than
            # re-probing every table against current connector state.
            direct_source_aliases = set()
            source_decisions = ReadThroughTableCache[
                DirectDiskgraphSourceDecision](
                    os.path.join(data_persist_dir, CACHE_DIR),
                    name=DIRECT_DISKGRAPH_SOURCE_DECISION,
                )
            for alias, table in all_graph_tables.items():
                if table.is_sql or not is_file_source_table(table):
                    continue
                decision = source_decisions.get(
                    table.fingerprint(),
                    str(table_data_versions[alias]),
                    cls=DirectDiskgraphSourceDecision,
                )
                if decision is None:
                    # Compatibility for refreshes completed before decisions
                    # were persisted. Only these legacy entries may re-probe.
                    compatibility_aliases.add(alias)
                elif decision.source is not None:
                    direct_source_aliases.add(alias)

        connectors: dict[str, Any] = {}
        deployment_environment = s3_subprocess_environment()
        ambient_s3_dataplane = (
            not deployment_environment
            and urlparse(
                deploy_env().data_plane_storage_root).scheme.lower() == 's3')
        environment: dict[str, Optional[str]] = (
            {} if ambient_s3_dataplane else scrubbed_s3_environment())
        environment.update(deployment_environment)
        direct_s3_environment: dict[str, str] | None = None
        prepared_connectors: set[str] = set()
        with ExitStack() as connector_stack:
            for alias, table in all_graph_tables.items():
                if table.is_sql:
                    continue
                if (alias not in direct_source_aliases
                        and alias not in compatibility_aliases):
                    continue
                connector_id = str(table.get_connector_id())
                if connector_id in prepared_connectors:
                    continue
                if connector_id not in connectors:
                    connector = get_connector_from_spec(
                        connector_specs[connector_id])
                    connector_stack.callback(_close_connector_quietly,
                                             connector)
                    connectors[connector_id] = connector
                connector = connectors[connector_id]
                if not isinstance(connector, FileConnector):
                    prepared_connectors.add(connector_id)
                    continue
                source_name = table.get_source_table_name()
                if (alias in compatibility_aliases
                        and not connector.uses_direct_diskgraph_source_for(
                            source_name)):
                    continue
                candidate = connector.diskgraph_subprocess_environment(
                    source_name)
                ambient_s3_candidate = (bool(candidate) and set(candidate) <= {
                    'AWS_REGION',
                    'AWS_DEFAULT_REGION',
                })
                if any(key.startswith(('AWS_', 'S3_'))
                       for key in candidate):
                    if (deployment_environment
                            and candidate != deployment_environment):
                        raise RuntimeError(
                            'Direct S3 source and deployment storage require '
                            'different connections; stage the source before '
                            'DiskGraph ingest.')
                    if (direct_s3_environment is not None
                            and direct_s3_environment != candidate):
                        raise RuntimeError(
                            'Direct S3 sources require different connector '
                            'connections in one DiskGraph job.')
                    direct_s3_environment = candidate
                    if not deployment_environment:
                        if ambient_s3_candidate:
                            # Keep only the attested renewable provider inputs.
                            environment.update(scrubbed_s3_environment())
                            for key in AWS_WORKLOAD_CREDENTIAL_ENVIRONMENT_KEYS:
                                value = os.environ.get(key)
                                if value is not None:
                                    environment[key] = value
                            # Instance-role refresh requires metadata access.
                            environment['AWS_EC2_METADATA_DISABLED'] = None
                            environment.update(
                                AWS_WORKLOAD_CREDENTIAL_PROFILE_ISOLATION_ENVIRONMENT)
                        else:
                            environment.update(scrubbed_s3_environment())
                for key, value in candidate.items():
                    existing = environment.get(key)
                    if existing is not None and existing != value:
                        raise RuntimeError(
                            'Direct object sources require conflicting '
                            f'subprocess values for {key}.')
                    environment[key] = value
                prepared_connectors.add(connector_id)
        return environment

    def restore_snapshot(
        self,
        handle: DiskgraphHandle,
        dataplane_root: str,
        *,
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> Optional[IngestResult]:
        """Bulk-copy a snapshot from its dataplane mirror to local SSD,
        then cross-check via ``verify_snapshot_integrity``.

        Copies the whole mirror with a plain ``fsutil.copy`` (measured
        faster on real GCS than planning the transfer from
        ``_MANIFEST.json``, since that manifest's only value was avoiding a
        serial directory walk and the walk is no longer serial). ``_SUCCESS``
        is copied last, on its own, after the rest of the tree.
        ``heartbeat``, if given, is invoked once before the transfer starts
        and then from a ticker thread for its duration, so a Temporal
        activity heartbeat timeout does not fire mid-restore.

        Returns ``None`` when the dataplane mirror is absent (no
        ``_SUCCESS`` at the expected dir). Raises
        ``DiskgraphCorruptionError`` if the mirror exists but the
        local copy fails ``verify_snapshot_integrity``.
        """
        from diskgraph import read_success_marker, verify_snapshot_integrity
        from kumo_core.diskgraph.snapshot import DiskgraphIngestResult

        from io_utils import fsutil

        dataplane_dir = self._dataplane_path(handle, dataplane_root)
        if not fsutil.exists(f'{dataplane_dir}/{_RESTORE_SUCCESS_FILENAME}'):
            return None
        output_dir = self._resolve(handle)
        _copy_dataplane_mirror(dataplane_dir, output_dir, heartbeat=heartbeat)
        verify_snapshot_integrity(output_dir)
        raw = read_success_marker(output_dir)
        # Explicit raise (not assert) so this survives ``python -O`` — a
        # stripped assert would fall through to ``json.loads(None)``.
        if raw is None:
            from statedb.utils.errors import DiskgraphError
            raise DiskgraphError(f'_SUCCESS missing at {output_dir}')
        dg = from_json_dict(json.loads(raw), DiskgraphIngestResult)
        return IngestResult(
            index_path=dg.index_path,
            col_stats_path=dg.col_stats_path,
            ingest_stats_path=dg.ingest_stats_path,
            kumo_stats_dir=dg.kumo_stats_dir,
        )

    def verify_integrity(self, handle: DiskgraphHandle) -> IngestResult:
        """Verify the local snapshot dir against its ``_MANIFEST.json``.

        Raises :class:`DiskgraphCorruptionError` on any deviation; returns the
        ``IngestResult`` from ``_SUCCESS``. Read-only — must not touch the
        marker (eviction is the pool daemon's job).
        """
        from diskgraph import read_success_marker, verify_snapshot_integrity
        from kumo_core.diskgraph.snapshot import DiskgraphIngestResult

        path = self._resolve(handle)
        verify_snapshot_integrity(path)
        raw = read_success_marker(path)
        # Explicit raise (not ``assert``) so this invariant survives
        # ``python -O`` — see ``restore_snapshot`` for the same pattern.
        if raw is None:
            from statedb.utils.errors import DiskgraphError
            raise DiskgraphError(
                f'verify_snapshot_integrity passed but _SUCCESS missing '
                f'at {path}')
        dg = from_json_dict(json.loads(raw), DiskgraphIngestResult)
        return IngestResult(
            index_path=dg.index_path,
            col_stats_path=dg.col_stats_path,
            ingest_stats_path=dg.ingest_stats_path,
            kumo_stats_dir=dg.kumo_stats_dir,
        )

    @blocking_method()
    def download_table_files(
        self,
        connector_specs: Any,
        table_data_versions: dict[str, str],
        all_graph_tables: Any,
        source_paths: dict[str, str],
    ) -> None:
        """Download source parquets to the per-snapshot tmp dir.

        Idempotent per table via a versioned ``_DOWNLOAD_SUCCESS`` inventory.
        OOM retries reuse a staged table only when its exact file identities
        and table-attempt binding still match. ``all_graph_tables`` maps table
        alias -> TableDefinition-shaped object; the alias keys
        ``source_paths`` / ``table_data_versions``.
        """
        import logging

        from io_utils import fsutil
        from rest.utils.driver_common import get_connector_from_spec

        log = logging.getLogger(__name__)
        connectors: dict = {}
        downloaded = skipped = 0
        for alias, local_dest in source_paths.items():
            table = all_graph_tables[alias]
            data_version = table_data_versions[alias]
            _remove_stale_download_attempts(local_dest)
            _recover_interrupted_download_swap(
                alias, table, data_version, local_dest)
            marker = os.path.join(local_dest, _DOWNLOAD_SUCCESS_FILENAME)
            if os.path.exists(marker):
                valid, reason = _validate_download_inventory(
                    marker, alias, table, data_version, local_dest)
                if valid:
                    skipped += 1
                    continue
                log.warning(
                    'download_table_files: rejecting staged source for %s: %s; '
                    'downloading a fresh owned copy', alias, reason)
            connector_id = table.get_connector_id()
            if connector_id not in connectors:
                spec = connector_specs.get(connector_id)
                if not spec:
                    log.warning(
                        'download_table_files: no connector spec for %s — skipping',
                        connector_id,
                    )
                    continue
                connectors[connector_id] = get_connector_from_spec(spec)
            connector = connectors[connector_id]
            parent = os.path.dirname(local_dest)
            fsutil.makedirs(parent, exist_ok=True)
            attempt_dir = tempfile.mkdtemp(
                prefix=f'.{os.path.basename(local_dest)}.download-',
                dir=parent,
            )
            try:
                connector.download(
                    table.get_source_table_name(),
                    table.fingerprint(),
                    str(data_version),
                    attempt_dir,
                )
                # Build and publish the marker inside the attempt directory.
                # Connector failures therefore leave the previous cache
                # untouched, while a crash before the final rename exposes no
                # reusable partial copy at ``local_dest``.
                payload = _download_inventory_payload(
                    alias, table, data_version, attempt_dir)
                _write_download_inventory_marker(
                    os.path.join(attempt_dir, _DOWNLOAD_SUCCESS_FILENAME),
                    payload,
                )
                _publish_download_attempt(
                    alias,
                    table,
                    data_version,
                    attempt_dir,
                    local_dest,
                )
            finally:
                if os.path.isdir(attempt_dir):
                    shutil.rmtree(attempt_dir)
            downloaded += 1
        log.info('download_table_files: downloaded=%d skipped=%d', downloaded,
                 skipped)

    @blocking_method()
    def has_resumable_ingest(self, handle: DiskgraphHandle) -> bool:
        """Return whether DiskGraph can resume this unpublished ingest."""
        from diskgraph import has_resumable_ingest

        root = self._resolve(handle)
        return (has_resumable_ingest(root)
                and self._has_valid_persisted_ingest_config(
                    handle,
                    check_local_paths=False,
                ))

    @blocking_method()
    def has_persisted_ingest_config(self, handle: DiskgraphHandle) -> bool:
        """Return whether an early retry can reuse an exact source config."""
        return self._has_valid_persisted_ingest_config(handle)

    def _has_valid_persisted_ingest_config(
        self,
        handle: DiskgraphHandle,
        *,
        check_local_paths: bool = True,
    ) -> bool:
        """Validate the non-secret exact source config used by all retries."""
        from kumo_core.diskgraph.runner import (
            _local_filesystem_path,
            persisted_ingest_config_path,
        )

        path = persisted_ingest_config_path(Path(self._resolve(handle)))
        try:
            config = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(config, dict):
            return False
        tables = config.get('tables')
        if not isinstance(tables, list):
            return False
        if not check_local_paths:
            return True
        for table in tables:
            source = table.get('source') if isinstance(table, dict) else None
            if (not isinstance(source, dict)
                    or source.get('type') not in ('parquet', 'csv')):
                continue
            paths = source.get('paths')
            if not isinstance(paths, list) or not paths:
                return False
            for source_path in paths:
                if not isinstance(source_path, str) or not source_path.strip():
                    return False
                try:
                    local_path = _local_filesystem_path(source_path)
                    if local_path is not None and not local_path.exists():
                        return False
                except (OSError, ValueError):
                    return False
        return True

    @blocking_method()
    def cleanup_ingest_sidecars(
        self,
        handle: DiskgraphHandle,
        *,
        preserve_config: bool = False,
    ) -> None:
        """Remove private ingest state, optionally retaining exact sources."""
        from kumo_core.diskgraph.runner import persisted_ingest_config_path

        if preserve_config:
            return
        root = self._resolve(handle)
        try:
            os.unlink(str(persisted_ingest_config_path(Path(root))))
        except FileNotFoundError:
            pass

    @blocking_method()
    def copy_ingest_stats_to_cache(
        self,
        handle: DiskgraphHandle,
        index_path: str,
        ingest_stats_path: str,
        *,
        tables: Any,
        table_data_versions: dict,
        has_edges: bool = True,
    ) -> Optional[dict[str, int]]:
        """Copy per-table ingest stats into the kumo graph-cache layout.

        Args:
            index_path: Path to the DG ingest index (from a prior
                :meth:`ingest` call's ``result.index_path``).
            tables: Iterable of graph tables whose stats should be cached.
            table_data_versions: Map of source table name to data
                version string used to key the cache entries.
        """
        from kumo.diskgraph.ingest_stats_cache import \
            copy_ingest_stats_to_backend_cache  # isort:skip
        return copy_ingest_stats_to_backend_cache(
            index_path=index_path,
            ingest_stats_path=ingest_stats_path,
            graph_key=handle.graph_key,
            graph_data_version=handle.graph_data_version,
            tables=tables,
            table_data_versions=table_data_versions,
            has_edges=has_edges,
        )

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def generate_train_table(
        self,
        handle: DiskgraphHandle,
        query: Any,
        timestamps: List[int],
        task_type: str,
        forecast_length: int,
        lag_timesteps: Optional[int],
        list_distinct_target_is_array: bool,
        include_frequency: bool,
        entity_time_col: Optional[str],
        output_path: str,
        split_config: Any,
        input_timestamp_unit: TimeUnit,
        target_stype: str,
        split_family: Optional[str] = None,
        downsampling: Optional[dict] = None,
        train_table_edges_config: Optional[dict] = None,
        index_path: Optional[str] = None,
        graph: Any = None,
    ) -> TrainTableResult:
        """Generate the training table for a PQL query against the snapshot.

        Args:
            handle: Snapshot identity the training table is generated against.
            query: Parsed PQL query (``PredictiveQuery`` or ``PredictiveQueryV2``).
            timestamps: Anchor timestamps (epoch seconds) at which training
                examples are sampled.
            task_type: Task-type identifier (regression, classification, link
                prediction, etc.) used to drive label/edge construction.
            forecast_length: Forecast horizon in time-steps for forecasting
                tasks; ignored for non-forecasting tasks.
            lag_timesteps: Optional lag window for forecasting tasks.
            list_distinct_target_is_array: Treat list-distinct targets as
                array-valued instead of scalar-list.
            include_frequency: Emit a frequency column alongside labels.
            entity_time_col: Optional per-entity time column name.
            output_path: Directory where the training table parquet is written.
            split_config: Train/val/test split specification.
            split_family: Split family identifier (random / temporal / etc.).
            downsampling: Optional downsampling configuration dict.
            train_table_edges_config: Optional fkey-edge bundle config.
            index_path: Override for the diskgraph index path; defaults to
                the handle's resolved path.
            graph: Optional graph object for label-side enrichment.

        Returns:
            Output parquet path and artifact-bundle manifest path.
        """
        import kumo.data.generic.diskgraph_train_table as train_table_mod

        out, manifest = (train_table_mod.generate_train_table_via_diskgraph(
            query=query,
            graph=graph,
            timestamps=timestamps,
            task_type=task_type,
            forecast_length=forecast_length,
            lag_timesteps=lag_timesteps,
            list_distinct_target_is_array=list_distinct_target_is_array,
            include_frequency=include_frequency,
            entity_time_col=entity_time_col,
            input_timestamp_unit=input_timestamp_unit,
            target_stype=target_stype,
            output_path=output_path,
            split_config=split_config,
            split_family=split_family,
            downsampling=downsampling,
            train_table_edges_config=train_table_edges_config,
            index_path=index_path or self._resolve(handle),
        ))
        return TrainTableResult(output_path=out, manifest_path=manifest)

    def generate_pred_table(
        self,
        handle: DiskgraphHandle,
        query: Any,
        task_type: Any,
        graph: Any,
        timestamps: List[int],
        forecast_length: int,
        lag_timesteps: Optional[int],
        use_yoy_labels: bool,
        list_distinct_target_is_array: bool,
        output_root: str,
        input_timestamp_unit: TimeUnit,
        target_stype: str,
        entity_time_col: Optional[str],
        target_output_root: Optional[str] = None,
        index_path: Optional[str] = None,
        lp_bp_layout: Optional[str] = None,
    ) -> PredTableResult:
        """Generate the batch-prediction table for a PQL query.

        Args:
            handle: Snapshot identity the prediction table is generated
                against.
            query: Parsed PQL query.
            task_type: Task-type identifier driving label/edge construction.
            graph: Graph object used for label-side enrichment.
            timestamps: Anchor timestamps (epoch seconds) at which prediction
                rows are emitted.
            forecast_length: Forecast horizon in time-steps for forecasting
                tasks.
            lag_timesteps: Optional lag window for forecasting tasks.
            use_yoy_labels: Emit year-over-year-style labels for forecasting.
            list_distinct_target_is_array: Treat list-distinct targets as
                array-valued.
            output_root: Directory where partitioned prediction-table parquet
                files are written.
            target_output_root: Optional override for target-table output.
            target_stype: Required kumo target stype (e.g. "categorical",
                "numerical", "multicategorical"); drives label-table
                col-stats shape on the inference side.
            index_path: Override for the diskgraph index path; defaults to
                the handle's resolved path.

        Returns:
            Total row count and number of output partitions.
        """
        import kumo.data.generic.diskgraph_batch_pred as batch_pred_mod

        rows, parts = (batch_pred_mod.generate_batch_pred_table_via_diskgraph(
            query=query,
            task_type=task_type,
            graph=graph,
            timestamps=timestamps,
            forecast_length=forecast_length,
            lag_timesteps=lag_timesteps,
            use_yoy_labels=use_yoy_labels,
            list_distinct_target_is_array=list_distinct_target_is_array,
            output_root=output_root,
            target_output_root=target_output_root,
            index_path=index_path or self._resolve(handle),
            input_timestamp_unit=input_timestamp_unit,
            target_stype=target_stype,
            entity_time_col=entity_time_col,
            lp_bp_layout=lp_bp_layout,
        ))
        return PredTableResult(total_rows=rows, total_partitions=parts)

    # ------------------------------------------------------------------
    # Task-dataset materialization
    # ------------------------------------------------------------------

    def materialize_task_dataset(
        self,
        handle: DiskgraphHandle,
        spec: Any,
    ) -> MaterializeResult:
        """Materialize a task dataset from TT-gen output: consume the wheel's
        ``task_dataset`` block and write the kumo-side manifests pointing at the
        ingested stage tables.
        """
        from io_utils import fsutil
        from kumo.diskgraph import diskgraph_native_tt_artifacts

        graph_index_id = f'{handle.graph_key}/{handle.graph_data_version}'
        graph_index_path = self._resolve(handle)

        # CONTRACT: post-ingest, the durable bundle ``execute_tt_ingest``
        # publishes is the SINGLE authoritative source: ``diskgraph_manifest
        # .json`` -> ``task_dataset``, ``stats.json`` -> ``parsed_stats``. The
        # gen ``manifest.json`` (``spec.tt_manifest_path``) is NOT read here --
        # its internal paths reference the gen pod's consumed scratch staging,
        # torn down before training. A missing ingest manifest means the
        # extension is incomplete/stale (NOT a cue to fall back to gen scratch);
        # the materialize activity re-ingests from retained staging first
        # (``_ttng_ensure_train_table_extension``), so we raise if still absent.
        manifest_dir = os.path.dirname(spec.tt_manifest_path)
        dg_manifest_path = os.path.join(manifest_dir, 'diskgraph_manifest.json')
        durable_stats_path = os.path.join(manifest_dir, 'stats.json')

        if not fsutil.exists(dg_manifest_path):
            raise RuntimeError(
                f'TT-NG ingest manifest missing at {dg_manifest_path}: the '
                'durable ingest extension is incomplete or stale. (Re)ingest '
                'the TT-NG extension from its persisted staging before '
                'materialization.')
        try:
            dg_manifest = json.loads(fsutil.read_binary(dg_manifest_path))
        except (ValueError, OSError) as ex:
            raise RuntimeError(
                f'TT-NG ingest manifest at {dg_manifest_path} is unreadable or '
                'corrupt; re-ingest the TT-NG extension.') from ex
        task_dataset: Any = (dg_manifest.get('task_dataset') if isinstance(
            dg_manifest, dict) else None)
        if not isinstance(task_dataset, dict):
            raise RuntimeError(
                f'TT-NG ingest manifest at {dg_manifest_path} is missing '
                '``task_dataset``; re-ingest the TT-NG extension.')

        _validate_ttng_ingest_binding(
            dg_manifest,
            task_dataset,
            graph_index_path=graph_index_path,
            tt_extension_path=manifest_dir,
            tt_id=os.path.basename(os.path.normpath(manifest_dir)),
        )
        spec = with_resolved_target_encoder_input(spec, dg_manifest,
                                                  manifest_dir)

        if not fsutil.exists(durable_stats_path):
            raise RuntimeError(
                f'TT-NG ingest stats sidecar missing at {durable_stats_path}: '
                'the durable ingest bundle is incomplete; re-ingest the TT-NG '
                'extension.')
        parsed_stats: Any = (diskgraph_native_tt_artifacts.parse_stats_artifact(
            durable_stats_path))

        # Stage row_counts + seed_times in ``task_dataset.stages`` (split.bin removed).
        # Seed times are int64-ns ``[min, max]`` pairs, emitted for temporal LP.
        wheel_stages = task_dataset.get('stages')
        if not isinstance(wheel_stages, dict) or not wheel_stages:
            raise RuntimeError('TT manifest task_dataset.stages empty')
        stage_counts: dict[str, int] = {}
        stage_seed_times: dict[str, tuple[int, int]] | None = {}
        for stage_name, info in wheel_stages.items():
            if stage_name == 'unused':
                continue
            if not isinstance(info, dict):
                continue
            rc = info.get('row_count')
            if isinstance(rc, int):
                stage_counts[stage_name] = rc
            st = info.get('seed_time')
            if (isinstance(st, (list, tuple)) and len(st) == 2
                    and all(isinstance(v, int) for v in st)):
                stage_seed_times[stage_name] = (int(st[0]), int(st[1]))
        if not stage_counts:
            raise RuntimeError(
                'TT manifest task_dataset.stages has no row_counts')
        if not stage_seed_times:
            stage_seed_times = None

        # The wheel sets ``columns.timestamp`` / ``columns.weight`` to
        # the column name when present, otherwise null.
        columns = task_dataset.get('columns') or {}
        has_time = bool(columns.get('timestamp'))
        has_weight = bool(columns.get('weight'))
        tt_extension_path = task_dataset.get('tt_extension_path')
        if not isinstance(tt_extension_path, str) or not tt_extension_path:
            raise RuntimeError(
                'TT manifest task_dataset block is malformed; expected '
                'non-empty tt_extension_path')
        stage_table_refs: dict[str, dict[str, Any]] = {}
        for stage in stage_counts:
            stage_info = wheel_stages.get(stage)
            if not isinstance(stage_info, dict):
                raise RuntimeError(
                    'TT manifest task_dataset.stages missing entry for '
                    f'stage={stage!r}')
            table_name = stage_info.get('table_name')
            index_path = stage_info.get('index_path')
            if not isinstance(table_name, str) or not table_name:
                raise RuntimeError(
                    f'task_dataset.stages[{stage!r}].table_name missing')
            if not isinstance(index_path, str) or not index_path:
                raise RuntimeError(
                    f'task_dataset.stages[{stage!r}].index_path missing')
            stage_table_refs[stage] = {
                'table_name': table_name,
                'index_path': index_path,
                'storage_kind': 'diskgraph_table',
            }
            binary_label_index = stage_info.get('binary_label_index')
            if isinstance(binary_label_index, dict):
                stage_table_refs[stage][
                    'binary_label_index'] = binary_label_index
            sample_weight_index = stage_info.get('sample_weight_index')
            if isinstance(sample_weight_index, dict):
                stage_table_refs[stage][
                    'sample_weight_index'] = sample_weight_index

        # Sidecar bundle (Kumo metadata index); downstream callers read stats.
        manifest_path = write_thin_manifest(
            spec,
            graph_index_id=graph_index_id,
            graph_index_path=graph_index_path,
            stats=parsed_stats or {},
            stage_counts=stage_counts,
            stage_seed_times=stage_seed_times,
            stage_table_refs=stage_table_refs,
            has_time=has_time,
            has_weight=has_weight,
        )
        bundle_dir = os.path.dirname(manifest_path)

        # Write the flat runtime manifest (wheel iterator input).
        # Per-stage variation is in ``stage_table_refs[stage]['table_name']``.
        runtime_manifest_path = write_flat_manifest(
            spec,
            bundle_dir=bundle_dir,
            graph_index_path=graph_index_path,
            stage_index_path=tt_extension_path,
            stage_counts=stage_counts,
            stage_table_refs=stage_table_refs,
            has_time=has_time,
            has_weight=has_weight,
        )

        return MaterializeResult(
            manifest_path=manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            bundle_dir=bundle_dir,
            storage_kind='diskgraph_table',
        )

    def materialize_prediction_task_dataset(
        self,
        handle: DiskgraphHandle,
        spec: Any,
        *,
        predict_staging_dir: str,
        predict_seed_time_ns: Optional[int] = None,
    ) -> MaterializeResult:
        """Materialize a TT-NG prediction task dataset (BP inference).

        The generator already ran ``execute_tt_gen`` into ``predict_staging_dir``;
        ingest it into a per-BP-job target extension.
        """
        graph_index_id = f'{handle.graph_key}/{handle.graph_data_version}'
        graph_index_path = self._resolve(handle)

        # `pred_` prefix keeps it distinct from the train TT under the same job id.
        from diskgraph import execute_tt_ingest

        tt_id = f'pred_{spec.job_id}'
        ingest_result = execute_tt_ingest(
            staging_dir=predict_staging_dir,
            index_path=graph_index_path,
            tt_id=tt_id,
            predict_stage_name='predict',
        )
        expected_extension_path = os.path.join(graph_index_path, 'train_tables',
                                               tt_id)
        td = ingest_result.get('task_dataset')
        if not isinstance(td, dict):
            raise RuntimeError(
                'execute_tt_ingest (predict) result missing '
                'task_dataset; re-ingest the prediction task dataset.')
        _validate_ttng_ingest_binding(
            ingest_result,
            td,
            graph_index_path=graph_index_path,
            tt_extension_path=expected_extension_path,
            tt_id=tt_id,
        )
        # Predict seeds are membership-verified like train seeds; warn (or
        # hard-fail via KUMO_TTNG_ORPHAN_FAIL_PCT) on references this graph
        # can't resolve — a silent skip here means missing prediction rows.
        from kumo.data.generic import diskgraph_staging
        diskgraph_staging.check_ingest_orphan_skips(ingest_result, 'predict')

        # The published ingest manifest is the sole authority for the
        # table name and row count. Do not guess values from gen output:
        # that would permit a stale/cross-graph extension to be consumed.
        td_stages = td['stages']
        predict_stage = td_stages.get('predict')
        if not isinstance(predict_stage, dict):
            raise RuntimeError(
                'TT-NG ingest manifest missing predict stage; re-ingest '
                'the prediction task dataset.')
        predict_table_name = predict_stage.get('table_name')
        predict_row_count = predict_stage.get('row_count')
        if not isinstance(predict_table_name, str) or not predict_table_name:
            raise RuntimeError(
                'TT-NG ingest manifest predict stage missing table_name; '
                're-ingest the prediction task dataset.')
        if (not isinstance(predict_row_count, int) or predict_row_count < 0):
            raise RuntimeError(
                'TT-NG ingest manifest predict stage has invalid row_count; '
                're-ingest the prediction task dataset.')
        tt_extension_path = td['tt_extension_path']

        stage_counts = {'predict': predict_row_count}
        stage_table_refs = {
            'predict': {
                'table_name': predict_table_name,
                'index_path': predict_stage['index_path'],
                'storage_kind': 'diskgraph_table',
            }
        }
        ext_root = tt_extension_path
        # Predict seed-time range = the anchor ``[t, t]`` so the RHS MIPS
        # NodeRange loader gets a non-None seed time. None for node/non-LP.
        stage_seed_times: dict[str, list[int]] | None = None
        if predict_seed_time_ns is not None:
            t = int(predict_seed_time_ns)
            stage_seed_times = {'predict': [t, t]}
        return self._write_prediction_bundle(
            spec,
            graph_index_id=graph_index_id,
            graph_index_path=graph_index_path,
            ext_root=ext_root,
            stage_counts=stage_counts,
            stage_table_refs=stage_table_refs,
            stage_seed_times=stage_seed_times,
        )

    def _write_prediction_bundle(
        self,
        spec: Any,
        *,
        graph_index_id: str,
        graph_index_path: str,
        ext_root: str,
        stage_counts: dict,
        stage_table_refs: dict,
        stage_seed_times: dict[str, list[int]] | None = None,
    ) -> MaterializeResult:
        """Write the thin + flat ``predict``-stage manifests for a BP task
        dataset. Shared by both ingest bridges so the bundle layout is identical.
        """
        manifest_path = write_thin_manifest(
            spec,
            graph_index_id=graph_index_id,
            graph_index_path=graph_index_path,
            stats={},
            stage_counts=stage_counts,
            stage_seed_times=stage_seed_times,
            stage_table_refs=stage_table_refs,
            has_time=True,
            has_weight=False,
            column_stats=False,
        )
        bundle_dir = os.path.dirname(manifest_path)

        runtime_manifest_path = write_flat_manifest(
            spec,
            bundle_dir=bundle_dir,
            graph_index_path=graph_index_path,
            stage_index_path=ext_root,
            stage_counts=stage_counts,
            stage_table_refs=stage_table_refs,
            has_time=True,
            has_weight=False,
        )

        return MaterializeResult(
            manifest_path=manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            bundle_dir=bundle_dir,
            storage_kind='diskgraph_table',
            predict_row_count=int(stage_counts.get('predict', 0)),
        )

    # ------------------------------------------------------------------
    # Stats + introspection
    # ------------------------------------------------------------------

    @blocking_method()
    def load_ingest_index_metadata(
        self,
        handle: DiskgraphHandle,
        *,
        index_path: Optional[str] = None,
    ) -> dict:
        """Return the parsed ``index_metadata.json`` from the snapshot.

        Args:
            handle: Snapshot identity to load metadata for.
            index_path: Override for the diskgraph index path; defaults to
                the handle's resolved path.

        Returns:
            Parsed JSON dict containing per-node-type counts, edge
            relations, and other ingest-time metadata.
        """
        from diskgraph import read_index_metadata as _read_index_metadata

        return json.loads(
            _read_index_metadata(index_path or self._resolve(handle)))

    def validate_manifest(self, manifest_path: str) -> Any:
        """Validate a diskgraph-native artifact manifest and return the
        parsed bundle.
        """
        from kumo.diskgraph.diskgraph_native_tt_artifacts import \
            get_validated_artifact_bundle  # isort:skip
        return get_validated_artifact_bundle(manifest_path)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def cleanup_staging(self, job_ids: List[str]) -> None:
        """Remove per-job staging directories under ``data_root``.

        Cleans up the ``rehydration_cache``, ``edge_persist``, and
        ``pred_artifact_cache`` subdirectories scoped to each job-id.
        Does not affect concurrent jobs because each job's staging is
        scoped under its own job-id subdirectory.

        Args:
            job_ids: List of job-id strings whose staging dirs should
                be removed. Empty / falsy entries are skipped.
        """
        data_root = self._config.data_root
        for job_id in job_ids:
            if not job_id:
                continue
            for subdir in (
                    'rehydration_cache',
                    'edge_persist',
                    'pred_artifact_cache',
            ):
                staging_dir = os.path.join(data_root, subdir, job_id)
                if os.path.isdir(staging_dir):
                    shutil.rmtree(staging_dir, ignore_errors=True)

    @blocking_method()
    def compute_col_stats_for_parquet(
        self,
        parquet_paths: List[str],
        kumo_column_specs: List[dict],
        data_version: str,
    ) -> Tuple[str, str]:
        """Compute Kumo-format column statistics over BP-output parquet.

        Reads the parquet once (row groups parallel) and derives both the full
        and min (truncated-CATEGORY_COUNTS) variants.

        Args:
            parquet_paths: Parquet files; row groups merged across all.
            kumo_column_specs: One dict per column (schema order) with
                ``name`` / ``stype`` / ``dtype`` (raw Kumo enum value strings);
                pybind maps them via ``map_kumo_stype`` / ``map_kumo_dtype``.
            data_version: Stamped into every ``kumo_col_stats_v1`` entry.

        Returns:
            ``(full_json, min_json)`` — the full map and its min variant.
        """
        from diskgraph import compute_col_stats_for_parquet
        return compute_col_stats_for_parquet(
            parquet_paths=parquet_paths,
            kumo_column_specs_json=json.dumps(kumo_column_specs),
            data_version=data_version,
        )


def _to_kumo_col_specs(table_def: Any) -> List[dict]:
    """Project ``TableDefinition.cols`` into the
    ``compute_col_stats_for_parquet`` shape (raw Stype/Dtype value strings;
    diskgraph translates them). Columns without supported resolved types are
    omitted, matching the normal statistics pipeline.
    """
    specs = []
    for col in table_def.cols:
        stype = col.stype.value if col.stype is not None else 'unsupported'
        dtype = col.dtype.value if col.dtype is not None else 'unsupported'
        if 'unsupported' in {stype, dtype}:
            continue
        spec = {
            'name': col.name,
            'stype': stype,
            'dtype': dtype,
        }
        timestamp_format = getattr(col, 'timestamp_format', None)
        timestamp_format = getattr(timestamp_format, 'value', timestamp_format)
        if timestamp_format in {'s', 'ms', 'us', 'ns'}:
            spec['timestamp_unit'] = timestamp_format
        elif timestamp_format is not None:
            spec['timestamp_format'] = timestamp_format
        specs.append(spec)
    return specs
