# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source staging is discarded after success, acquisition failure, or ingest failure."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kumo.engine.graph import Graph


def _fake_table(dest_writes: bool = True, credentials=None):
    def prepare_ingest(connector, table_name, dest_dir, *, dtypes):
        if dest_writes:
            Path(dest_dir, 'part.parquet').write_bytes(b'data')
        return SimpleNamespace(credentials=credentials, version='v1',
                               source={'path': dest_dir})

    return SimpleNamespace(
        definition=SimpleNamespace(cols=[]),
        connector=SimpleNamespace(source_backend=SimpleNamespace(
            prepare_ingest=prepare_ingest)),
        table_name='t',
    )


def _patch_diskgraph_success(monkeypatch, tmp_path):
    monkeypatch.setattr(Graph, '_graph_config', {'name': 'g'})
    monkeypatch.setattr(
        'kumo_core.diskgraph.snapshot.LocalSnapshotStore', lambda config:
        SimpleNamespace(path=lambda handle: str(tmp_path / 'output')))
    monkeypatch.setattr('kumo_core.diskgraph.snapshot.DiskgraphSnapshotHandle',
                        lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr('kumo_core.diskgraph.snapshot.check_diskgraph_output',
                        lambda output_dir: None)
    monkeypatch.setattr(
        'kumo_core.diskgraph.snapshot.DiskgraphIngestResult.for_index',
        staticmethod(lambda output_dir: SimpleNamespace(to_json=lambda: '{}')))
    monkeypatch.setattr(
        'kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
        lambda graph_config, spec: SimpleNamespace(index_path='idx'))


def test_snapshot_discards_staging_dir_on_success(tmp_path, monkeypatch):
    _patch_diskgraph_success(monkeypatch, tmp_path)
    persist_dir = tmp_path / 'staging'
    graph = Graph(tables=[_fake_table()])

    result = graph._snapshot_no_lock('graph-key', persist_dir)

    assert result.index_path == Path('idx')
    assert not persist_dir.exists()


def test_snapshot_discards_staging_dir_on_partial_stage_failure(
        tmp_path, monkeypatch):
    _patch_diskgraph_success(monkeypatch, tmp_path)
    persist_dir = tmp_path / 'staging'
    # Two tables: the first stages successfully, the second's
    # `prepare_ingest` reports scoped credentials, which `_snapshot_no_lock`
    # doesn't support and raises for -- after the first table has already
    # written into `persist_dir`.
    tables = [
        _fake_table(),
        SimpleNamespace(
            **{
                **vars(_fake_table(credentials=object())),
                'table_name': 't2',
            }),
    ]
    graph = Graph(tables=tables)

    with pytest.raises(NotImplementedError, match='credentials'):
        graph._snapshot_no_lock('graph-key', persist_dir)

    assert not persist_dir.exists()


def test_snapshot_discards_staging_dir_on_ingest_failure(tmp_path, monkeypatch):
    _patch_diskgraph_success(monkeypatch, tmp_path)
    monkeypatch.setattr(
        'kumo_core.diskgraph.ingest.execute_diskgraph_ingest',
        lambda graph_config, spec:
        (_ for _ in ()).throw(RuntimeError('ingest failed')))
    persist_dir = tmp_path / 'staging'
    graph = Graph(tables=[_fake_table()])

    with pytest.raises(RuntimeError, match='ingest failed'):
        graph._snapshot_no_lock('graph-key', persist_dir)

    assert not persist_dir.exists()
