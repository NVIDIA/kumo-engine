# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir, scoped_scratch_dir


@pytest.fixture
def source_graph(tmp_path, monkeypatch):
    from kumo import settings

    previous = settings.scratch_dir()
    monkeypatch.setattr('kumo.engine._scratch._root', None)
    kumo.configure(scratch_dir=tmp_path / 'producer')
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(pa.table({'id': [1, 2, 3], 'value': [1., 2., 3.]}),
                   data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    graph = kumo.Graph({'events': kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='id')})
    try:
        yield graph
    finally:
        kumo.configure(scratch_dir=previous)


def test_graph_captures_root_at_construction_and_reuses_it(
        tmp_path, source_graph, monkeypatch):
    import kumo_core.diskgraph.ingest as ingest

    producer = tmp_path / 'producer'
    other = tmp_path / 'other'
    execute = ingest.execute_diskgraph_ingest
    outputs = []

    def record(graph_config, spec):
        outputs.append(Path(spec.output_dir))
        assert Path(spec.data_persist_dir).is_relative_to(outputs[-1].parents[1])
        return execute(graph_config, spec)

    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', record)
    kumo.configure(scratch_dir=other)
    first = source_graph.snapshot()
    assert first.index_path.is_relative_to(producer / 'indexes')
    assert not other.exists()
    assert source_graph.snapshot() == first
    assert len(outputs) == 1
    assert get_scratch_dir() == other

    new_graph = replace(source_graph)
    second = new_graph.snapshot()
    assert second.id == first.id
    assert second.index_path.is_relative_to(other / 'indexes')
    assert second.index_path != first.index_path
    assert len(outputs) == 2
    assert first.index_path.is_dir()


def test_callback_configuration_cannot_split_snapshot_paths(
        tmp_path, source_graph, monkeypatch):
    import kumo_core.diskgraph.ingest as ingest

    producer = tmp_path / 'producer'
    other = tmp_path / 'other'
    execute = ingest.execute_diskgraph_ingest
    validations = []
    validate = kumo.FileConnector.validate_scratch_dir

    def validate_root(connector, path):
        validations.append(path)
        return validate(connector, path)

    def execute_in_root(graph_config, spec):
        assert Path(spec.output_dir).is_relative_to(producer)
        assert Path(spec.data_persist_dir).is_relative_to(producer)
        assert (producer / 'snapshots' / source_graph.id /
                'source_tables.lock').is_file()
        assert not other.exists()
        return execute(graph_config, spec)

    def configure_other(event):
        if event.name == 'snapshot_started':
            kumo.configure(scratch_dir=other)

    monkeypatch.setattr(kumo.FileConnector, 'validate_scratch_dir', validate_root)
    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', execute_in_root)
    first = source_graph.snapshot(callbacks=[configure_other])
    assert validations == [producer]
    assert first.index_path.is_relative_to(producer / 'indexes')
    assert source_graph.snapshot() == first
    assert get_scratch_dir() == other
    assert not other.exists()


def test_failed_snapshot_restores_the_callers_root(
        tmp_path, source_graph, monkeypatch):
    import kumo_core.diskgraph.ingest as ingest

    other = tmp_path / 'other'

    def configure_other(event):
        if event.name == 'snapshot_started':
            kumo.configure(scratch_dir=other)

    def fail_ingest(graph_config, spec):
        assert Path(spec.output_dir).is_relative_to(tmp_path / 'producer')
        raise RuntimeError('ingest interrupted')

    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest', fail_ingest)
    caller_root = tmp_path / 'caller'
    with scoped_scratch_dir(caller_root):
        with pytest.raises(RuntimeError, match='ingest interrupted'):
            source_graph.snapshot(callbacks=[configure_other])
        assert get_scratch_dir() == caller_root
    assert get_scratch_dir() == other
    assert not caller_root.exists()
    assert not other.exists()



def test_snapshot_transfer_workers_keep_the_callers_spool_root(
        tmp_path, monkeypatch):
    from kumo_core.chunked_fs import ChunkedFileHandler
    from kumo_core.sources import object_store
    from pyarrow.fs import PyFileSystem

    from kumo import settings
    from kumo.engine import _snapshot_io

    class ChunkedExport(ChunkedFileHandler):
        def __init__(self):
            self.objects = {}
            self.spooled = []

        def get_type_name(self):
            return 'test-snapshot-export'

        def _size(self, path):
            data = self.objects.get(path)
            return None if data is None else len(data)

        def _list(self, directory):
            return {path: len(data) for path, data in self.objects.items()
                    if path.startswith(directory + '/')}

        def _put(self, local, path):
            self.spooled.append(Path(local).parent.parent)
            self.objects[path] = Path(local).read_bytes()

    previous = settings.scratch_dir()
    producer, other = tmp_path / 'producer', tmp_path / 'other'
    handler = ChunkedExport()
    filesystem = PyFileSystem(handler)
    monkeypatch.setattr(_snapshot_io, '_location',
                        lambda connector, path: (filesystem, path))
    monkeypatch.setattr(object_store, 'copies', lambda items: nullcontext(2))
    together = threading.Barrier(2)
    worker_state = ContextVar('snapshot-transfer-test', default='outside')
    token = worker_state.set('caller')

    def upload(filesystem, item, stop):
        assert worker_state.get() == 'caller'
        worker_state.set(item)
        with filesystem.open_output_stream(f'exports/{item}') as writer:
            together.wait(timeout=10)
            writer.write(b'abc')
        return item

    try:
        kumo.configure(scratch_dir=other)
        with scoped_scratch_dir(producer):
            assert _snapshot_io._transfer(
                None, 'exports', [0, 1], [3, 3], upload, 'store') == [0, 1]
            assert worker_state.get() == 'caller'
        assert get_scratch_dir() == other
    finally:
        worker_state.reset(token)
        kumo.configure(scratch_dir=previous)
    assert handler.objects == {'exports/0': b'abc', 'exports/1': b'abc'}
    assert handler.spooled == [producer / 'spool'] * 2
    assert not list((producer / 'spool').iterdir())
    assert not other.exists()
