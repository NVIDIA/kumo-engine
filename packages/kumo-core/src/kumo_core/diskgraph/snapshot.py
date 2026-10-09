# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable local snapshot identity, marker, verification, and restore logic."""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from kumoapi.plugin import SnapshotStore, _Handle

logger = logging.getLogger(__name__)


class DiskgraphSnapshotError(RuntimeError):
    """A snapshot marker or integrity invariant was violated."""


@dataclass(frozen=True)
class DiskgraphLocalConfig:
    data_root: str


@dataclass(frozen=True)
class DiskgraphSnapshotHandle:
    graph_key: str
    graph_data_version: str


@dataclass(frozen=True)
class DiskgraphIngestResult:
    """Stable on-disk shape persisted in a snapshot's ``_SUCCESS`` marker."""

    index_path: str
    col_stats_path: str
    ingest_stats_path: str
    kumo_stats_dir: str
    pkey_duplicate_counts: dict[str, int] | None = None

    @classmethod
    def for_index(cls, index_path: str | Path) -> "DiskgraphIngestResult":
        root = str(index_path)
        return cls(
            index_path=root,
            col_stats_path=os.path.join(root, "col_stats.json"),
            ingest_stats_path=os.path.join(root, "ingest_stats.json"),
            kumo_stats_dir=os.path.join(root, "kumo_col_stats"),
        )

    def to_json(self) -> str:
        payload = asdict(self)
        payload.pop("pkey_duplicate_counts", None)
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "DiskgraphIngestResult":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("DiskGraph success marker must contain an object")
        return cls(
            index_path=str(payload["index_path"]),
            col_stats_path=str(payload["col_stats_path"]),
            ingest_stats_path=str(payload["ingest_stats_path"]),
            kumo_stats_dir=str(payload["kumo_stats_dir"]),
            pkey_duplicate_counts=payload.get("pkey_duplicate_counts"),
        )

    def missing_paths(self) -> list[str]:
        missing = [
            path for path in (self.col_stats_path, self.ingest_stats_path)
            if not os.path.isfile(path)
        ]
        if not os.path.isdir(self.kumo_stats_dir):
            missing.append(self.kumo_stats_dir)
        return missing


def _to_diskgraph_object_store_path(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme not in ("abfs", "abfss") or "@" not in parsed.netloc:
        return path
    container, account_host = parsed.netloc.split("@", 1)
    account = account_host.split(".", 1)[0]
    key = parsed.path.lstrip("/")
    if not container or not account:
        return path
    return (f"az://{account}/{container}/{key}"
            if key else f"az://{account}/{container}")


class LocalSnapshotStore:
    """Manage reconstructable DiskGraph snapshots under an explicit root."""
    def __init__(self, config: DiskgraphLocalConfig) -> None:
        self.config = config

    def path(self, handle: DiskgraphSnapshotHandle) -> str:
        return os.path.join(
            self.config.data_root,
            handle.graph_key,
            handle.graph_data_version,
        )

    @staticmethod
    def durable_path(handle: DiskgraphSnapshotHandle, durable_root: str) -> str:
        return os.path.join(
            durable_root,
            "diskgraph_snapshots",
            handle.graph_key,
            handle.graph_data_version,
        )

    @classmethod
    def diskgraph_durable_path(cls, handle: DiskgraphSnapshotHandle,
                               durable_root: str) -> str:
        return _to_diskgraph_object_store_path(
            cls.durable_path(handle, durable_root))

    def predicted_result(
            self, handle: DiskgraphSnapshotHandle) -> DiskgraphIngestResult:
        return DiskgraphIngestResult.for_index(self.path(handle))

    def list_ready(self) -> list[DiskgraphSnapshotHandle]:
        ready: list[DiskgraphSnapshotHandle] = []
        try:
            graph_keys = os.listdir(self.config.data_root)
        except OSError:
            return ready
        for graph_key in graph_keys:
            graph_dir = os.path.join(self.config.data_root, graph_key)
            if not os.path.isdir(graph_dir):
                continue
            try:
                versions = os.listdir(graph_dir)
            except OSError:
                continue
            for version in versions:
                snapshot_dir = os.path.join(graph_dir, version)
                if not os.path.isdir(snapshot_dir):
                    continue
                try:
                    result = check_diskgraph_output(snapshot_dir)
                except DiskgraphSnapshotError:
                    logger.warning("Skipping invalid snapshot at %s",
                                   snapshot_dir)
                    continue
                if result is not None:
                    ready.append(DiskgraphSnapshotHandle(graph_key, version))
        return ready

    def verify(self, handle: DiskgraphSnapshotHandle) -> DiskgraphIngestResult:
        from diskgraph import read_success_marker, verify_snapshot_integrity

        path = self.path(handle)
        verify_snapshot_integrity(path)
        marker = read_success_marker(path)
        if marker is None:
            raise DiskgraphSnapshotError(
                "snapshot integrity passed but _SUCCESS is missing at " + path)
        validate_cached_source_times(path)
        return DiskgraphIngestResult.from_json(marker)

    def restore(
        self,
        handle: DiskgraphSnapshotHandle,
        durable_root: str,
        *,
        exists: Callable[[str], bool],
        copy_tree: Callable[[str, str], None],
    ) -> DiskgraphIngestResult | None:
        source = self.durable_path(handle, durable_root)
        if not exists(source + "/_SUCCESS"):
            return None
        copy_tree(source, self.path(handle))
        return self.verify(handle)

    def checkpoint(
        self,
        handle: DiskgraphSnapshotHandle,
        durable_root: str,
        *,
        exists: Callable[[str], bool],
        copy_tree: Callable[[str, str], None],
        replace_tree: Callable[[str, str], None],
    ) -> str:
        """Publish a verified local snapshot to durable storage atomically.

        Filesystem operations are injected so the same lifecycle works for
        Unity Catalog Volumes, test directories, and future object stores.
        """
        self.verify(handle)
        source = self.path(handle)
        target = self.durable_path(handle, durable_root)
        if exists(target + "/_SUCCESS"):
            return target
        staging = f"{target}.tmp-{uuid.uuid4().hex}"
        copy_tree(source, staging)
        if not exists(staging + "/_SUCCESS"):
            raise DiskgraphSnapshotError(
                "checkpoint copy completed without _SUCCESS at " + staging)
        replace_tree(staging, target)
        if not exists(target + "/_SUCCESS"):
            raise DiskgraphSnapshotError(
                "checkpoint publish completed without _SUCCESS at " + target)
        return target


def rebased(store: SnapshotStore[_Handle],
            handle: _Handle) -> DiskgraphIngestResult:
    r"""Returns a restored snapshot's result, addressed where it now sits.

    A ``_SUCCESS`` marker records *absolute* paths, fixed when the index was
    built. Restoring copies the index into this process's scratch, so a marker
    written under a different scratch root -- another container, another
    machine, the same graph ingested by someone else -- comes back naming a
    directory that does not exist here. What is stale is only where the files
    are said to be, not the files: every path in the marker is derived from
    the index root, which is exactly what
    :meth:`DiskgraphIngestResult.for_index` reconstructs. So the result is
    rebuilt against the local root rather than trusted as written.
    ``pkey_duplicate_counts`` is not lost with it: the marker never carries
    that field, since ``to_json`` drops it before writing.

    Every artifact location does this after restoring, whichever durable store
    it publishes to, so it lives beside the marker it re-reads rather than in
    any one of them.

    Raises:
        ValueError: If the restored copy is missing a file the marker
            promises, which means the durable copy is incomplete rather than
            merely addressed from elsewhere.
    """
    result = DiskgraphIngestResult.for_index(store.path(handle))
    missing = result.missing_paths()
    if missing:
        raise ValueError(
            f"Restored snapshot at {result.index_path} is missing "
            f"{', '.join(sorted(missing))}. The published copy is incomplete; "
            f"delete it so the next run ingests instead of restoring.")
    return result


def validate_cached_source_times(output_dir: str) -> None:
    """Reject cached required-time tables with no retained node rows.

    Parse failures describe discarded source rows, not invalid retained data.
    No source data is rescanned or cached artifacts modified here.
    """
    metadata_path = Path(output_dir) / 'index_metadata.json'
    if not metadata_path.is_file():
        return  # Structural/missing-artifact validation is a separate check.
    metadata = json.loads(metadata_path.read_text())
    for table in metadata.get('node_types', []):
        column = table.get('create_time_col')
        if not column:
            continue
        shards = table.get('shards', [])
        if not any(s['max_id'] >= s['min_id'] for s in shards):
            raise DiskgraphSnapshotError(
                f"Cached graph table {table['name']!r} has no usable rows "
                f"for {column!r}; correct the source and rebuild the snapshot")


def check_diskgraph_output(output_dir: str) -> DiskgraphIngestResult | None:
    """Return a valid marker result and best-effort touch it for LRU order."""

    from diskgraph import read_success_marker, touch_success_marker

    raw = read_success_marker(output_dir)
    if raw is None:
        return None
    result = DiskgraphIngestResult.from_json(raw)
    validate_cached_source_times(output_dir)
    logger.info("Valid diskgraph output found at %s", output_dir)
    try:
        touch_success_marker(output_dir)
    except Exception:
        logger.warning(
            "Failed to update _SUCCESS mtime at %s — LRU eviction order may "
            "be inaccurate for this index",
            output_dir,
            exc_info=True,
        )
    return result


def write_diskgraph_success_marker(output_dir: str,
                                   result: DiskgraphIngestResult) -> None:
    from diskgraph import write_success_marker

    write_success_marker(output_dir, result.to_json())
    logger.info("Wrote _SUCCESS marker to %s", output_dir)
