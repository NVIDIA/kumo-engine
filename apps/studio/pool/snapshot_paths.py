# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared snapshot path helpers."""
import posixpath
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.parse import quote

from pool.dataclasses import SnapshotID
from pool.errors import EnsureFailedSource


class NormalizedDiskgraphSnapshotId(NamedTuple):
    graph_fpid: str
    graph_data_version: str

    @property
    def graph_key(self) -> str:
        return f'NA-{self.graph_fpid}'


def dataplane_snapshot_path(data_plane_storage_root: str,
                            snapshot_id: str) -> str:
    """Return the L2 snapshot payload prefix."""
    normalized = normalize_diskgraph_snapshot_id(snapshot_id)
    if normalized is not None:
        return posixpath.join(
            data_plane_storage_root,
            'diskgraph_snapshots',
            normalized.graph_key,
            normalized.graph_data_version,
        )
    snapshot_segment = quote(snapshot_id, safe='')
    return posixpath.join(data_plane_storage_root, 'snapshots',
                          snapshot_segment)


def normalize_diskgraph_snapshot_id(
        snapshot_id: SnapshotID) -> Optional[NormalizedDiskgraphSnapshotId]:
    """Parse a diskgraph snapshot id into fpid + data version."""
    from statedb.config.common import DataSnapshot
    try:
        canonical = DataSnapshot.GRAPH.canonicalize(snapshot_id)
        fpid, version = DataSnapshot.GRAPH.parse_id(canonical)
    except Exception:
        return None
    return NormalizedDiskgraphSnapshotId(
        graph_fpid=str(fpid),
        graph_data_version=version,
    )


def snapshot_relative_path(snapshot_id: SnapshotID) -> Path:
    normalized = normalize_diskgraph_snapshot_id(snapshot_id)
    if normalized is None:
        return Path(snapshot_id)
    return Path(normalized.graph_key) / normalized.graph_data_version


def require_normalized_diskgraph_snapshot_id(
        snapshot_id: SnapshotID) -> NormalizedDiskgraphSnapshotId:
    normalized = normalize_diskgraph_snapshot_id(snapshot_id)
    if normalized is None:
        raise EnsureFailedSource(
            f'Fresh Diskgraph ingest requires a graph snapshot id: '
            f'{snapshot_id}')
    return normalized


def dataplane_root_from_snapshot_path(
    dataplane_path: Optional[str],
    graph_key: str,
    graph_data_version: str,
) -> str:
    if dataplane_path is None:
        raise EnsureFailedSource(
            'Fresh Diskgraph ingest requires a dataplane snapshot path.')
    normalized = dataplane_path.rstrip('/')
    candidate_keys = [graph_key]
    if graph_key.startswith(('NA-', 'graph-')):
        candidate_keys.append(graph_key.split('-', 1)[1])
    for candidate_key in candidate_keys:
        suffix = (
            f'diskgraph_snapshots/{candidate_key}/{graph_data_version}')
        if normalized == suffix:
            return ''
        if normalized.endswith(f'/{suffix}'):
            return normalized[:-(len(suffix) + 1)] or '/'
    raise EnsureFailedSource(
        'Dataplane snapshot path does not match Diskgraph layout: '
        f'{dataplane_path}')


def normalized_diskgraph_snapshot_id_from_path(
        graph_key: str, graph_data_version: str) -> Optional[SnapshotID]:
    if (not graph_data_version or
            not graph_key.startswith(('graph-', 'NA-'))):
        return None
    try:
        from statedb.config.common import DataSnapshot
        return DataSnapshot.GRAPH.canonicalize(
            f'{graph_key}@{graph_data_version}')
    except Exception:
        return None
