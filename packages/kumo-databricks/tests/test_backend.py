# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from kumo_core.sources.errors import (
    PermanentSourceError,
    UnsupportedReadModeError,
)
from kumo_core.sources.types import ReadMode, StagedTable
from kumo_databricks.backend import DatabricksSourceBackend


@pytest.mark.parametrize("rows", [0, 3])
def test_zero_row_read_is_an_empty_source(rows, tmp_path, monkeypatch):
    part = tmp_path / "part-0.parquet"
    part.write_bytes(b"schema only")
    staged = StagedTable(directory=str(tmp_path), paths=(str(part), ),
                         version="v1", source_size=1,
                         mode_used=ReadMode.STREAMED, num_rows=rows)
    monkeypatch.setattr("kumo_core.sources.prepare.read_table",
                        lambda *args, **kwargs: staged)
    backend = DatabricksSourceBackend()
    for name in ("_declared", "_reader"):
        monkeypatch.setattr(backend, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(backend, "_bare", lambda connector, table: table)
    monkeypatch.setattr(backend, "_connected",
                        lambda connector: nullcontext(None))
    connector = SimpleNamespace(id="c")

    def prepare():
        return backend.prepare_ingest(connector, "t", str(tmp_path), dtypes={})

    if rows:
        assert prepare().source["paths"] == [str(part)]
    else:
        with pytest.raises(PermanentSourceError, match="no rows"):
            prepare()


@pytest.mark.parametrize('directory', [False, True])
def test_volume_parquet_ingest_preserves_training_rows(directory, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fakes import FakeVolume, FakeWorkspaceClient

    path = '/Volumes/kumo_test_catalogue/kumo_test/test_stage/training_table'
    table = pa.table({
        'ENTITY1': [1, 2, 3],
        'TARGET1': [True, False, True],
        'SPLIT': [0, 1, 2]
    })
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    body = sink.getvalue().to_pybytes()
    source = path + '/part-00000.parquet' if directory else path + '.parquet'
    volume = FakeVolume({source: body})
    if directory:
        volume.objects[path + '/_SUCCESS'] = b'complete'
        volume.objects[path + '/_provenance.json'] = b'{}'
        volume.objects[path + '/unrelated.csv'] = b'ignore,this'
    backend = DatabricksSourceBackend()
    backend._client = lambda connector: FakeWorkspaceClient(volume)
    requested = path if directory else source
    prepared = backend.prepare_ingest(SimpleNamespace(), requested,
                                      str(tmp_path / 'staged'))
    parts = prepared.source['paths']
    assert len(parts) == 1
    assert pq.read_table(parts[0]).equals(table)
    from pathlib import Path

    assert Path(parts[0]).read_bytes() == body
    assert prepared.version == backend.get_data_version(SimpleNamespace(),
                                                        requested)
    assert prepared.credentials is None
    with pytest.raises(UnsupportedReadModeError, match='SQL'):
        backend.prepare_ingest(SimpleNamespace(), requested,
                               str(tmp_path / 'sql'), sql='SELECT 1')


@pytest.mark.parametrize('mode', [ReadMode.STREAMED, 'streamed'])
def test_volume_ingest_honors_projection_and_explicit_read_mode(tmp_path, mode):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fakes import FakeVolume, FakeWorkspaceClient
    from kumoapi.typing import Dtype

    path = '/Volumes/kumo_test_catalogue/kumo_test/test_stage/rows.parquet'
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({'ENTITY1': [1, 2], 'SPLIT': [0, 1]}), sink)
    volume = FakeVolume({path: sink.getvalue().to_pybytes()})
    backend = DatabricksSourceBackend(mode=mode)
    backend._client = lambda connector: FakeWorkspaceClient(volume)
    prepared = backend.prepare_ingest(SimpleNamespace(), path, str(tmp_path),
                                      dtypes={'ENTITY1': Dtype.int})
    assert pq.read_table(prepared.source['paths'][0]).to_pydict() == {
        'ENTITY1': [1, 2]
    }
    for refused in (ReadMode.STAGED, 'staged'):
        backend.mode = refused
        with pytest.raises(UnsupportedReadModeError):
            backend.prepare_ingest(SimpleNamespace(), path, str(tmp_path / 'mode'))


@pytest.mark.parametrize('path', [
    '/Volumes/kumo_test_catalogue/kumo_test/test_stage/../rows',
    '/Volumes/kumo_test_catalogue/kumo_test/test_stage//rows',
])
def test_invalid_volume_coordinate_never_opens_a_client(path, tmp_path):
    backend = DatabricksSourceBackend()
    backend._client = lambda connector: pytest.fail('Invalid path opened client'
                                                    )
    from kumoapi.data_source import DatabricksConnectorResourceConfig

    connector = SimpleNamespace(id='test', config=DatabricksConnectorResourceConfig(
        catalog='kumo_test_catalogue', schema_name='kumo_test',
        warehouse_id='KUMO_TEST_WH'))
    with pytest.raises(ValueError):
        backend.prepare_ingest(connector, path, str(tmp_path))
