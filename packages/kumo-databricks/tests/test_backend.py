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
        return backend.prepare_ingest(connector, "t", str(tmp_path),
                                      dtypes={})

    if rows:
        assert prepare().source["paths"] == [str(part)]
    else:
        with pytest.raises(PermanentSourceError, match="no rows"):
            prepare()


@pytest.fixture
def volume_source():
    from fakes import FakeVolume, FakeWorkspaceClient
    from kumoapi.data_source import DatabricksConnectorResourceConfig

    volume = FakeVolume()
    backend = DatabricksSourceBackend()
    backend._client = lambda connector: FakeWorkspaceClient(volume)
    connector = SimpleNamespace(
        id='example', config=DatabricksConnectorResourceConfig(
            catalog='example_catalog', schema_name='example_schema',
            warehouse_id='example-warehouse'))
    return backend, volume, connector, '/Volumes/example_catalog/example_schema/example_volume/source'


def parquet(values, dtype=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = pa.BufferOutputStream()
    pq.write_table(
        pa.table({
            'ENTITY1': pa.array(values, type=dtype or pa.int64()),
            'SPLIT': pa.array(list(range(len(values))), type=pa.int64())
        }), output)
    return output.getvalue().to_pybytes()


@pytest.mark.parametrize('directory', [False, True])
def test_volume_parquet_ingest_preserves_rows_and_metadata(
        volume_source, tmp_path, directory):
    from pathlib import Path

    import pyarrow.parquet as pq

    backend, volume, connector, root = volume_source
    path = root + '/part.parquet' if directory else root + '.parquet'
    volume.objects[path] = body = parquet([1, 2, 3])
    if directory:
        volume.objects[root + '/_SUCCESS'] = b'complete'
        volume.objects[root + '/_temporary/ignored.parquet'] = parquet([999])
    selected = root if directory else path
    before = backend.get_data_version(connector, selected)
    prepared = backend.prepare_ingest(connector, selected, str(tmp_path))
    assert len(prepared.source['paths']) == 1
    assert Path(prepared.source['paths'][0]).read_bytes() == body
    assert pq.read_table(prepared.source['paths'][0]).to_pydict() == {
        'ENTITY1': [1, 2, 3],
        'SPLIT': [0, 1, 2]
    }
    assert prepared.version == before == backend.get_data_version(
        connector, selected)
    assert before != '-1' and before.kind == 'listing'
    with pytest.raises(UnsupportedReadModeError, match='SQL'):
        backend.prepare_ingest(connector, selected, str(tmp_path / 'sql'),
                               sql='SELECT 1')


@pytest.mark.parametrize('directory', [False, True])
@pytest.mark.parametrize('values', [[3, 4], [10, 20, 30]])
def test_volume_rewrite_changes_version_even_with_same_size(
        volume_source, directory, values):
    backend, volume, connector, root = volume_source
    path = root + '/part.parquet' if directory else root + '.parquet'
    volume.objects[path] = original = parquet([1, 2])
    selected = root if directory else path
    before = backend.get_data_version(connector, selected)
    volume.objects[path] = replacement = parquet(values)
    if len(values) == 2:
        assert len(original) == len(replacement)
    volume.modified[path] = 1_700_000_000_001
    assert backend.get_data_version(connector, selected) != before


@pytest.mark.parametrize('dtypes', [None, {}])
@pytest.mark.parametrize('empty', ['file', 'folder'])
def test_empty_volume_input_has_permanent_no_rows_error(volume_source, tmp_path,
                                                        dtypes, empty):
    backend, volume, connector, root = volume_source
    if empty == 'file':
        root += '.parquet'
        volume.objects[root] = parquet([])
    else:
        volume.directories.add(root)
    with pytest.raises(PermanentSourceError, match='no rows'):
        backend.prepare_ingest(connector, root, str(tmp_path), dtypes=dtypes)


@pytest.mark.parametrize('mode', [ReadMode.STREAMED, 'streamed'])
def test_volume_uses_shared_projection_and_mode_validation(
        volume_source, tmp_path, mode):
    import pyarrow.parquet as pq
    from kumoapi.typing import Dtype

    backend, volume, connector, root = volume_source
    root += '.parquet'
    volume.objects[root] = parquet([1, 2])
    backend.mode = mode
    prepared = backend.prepare_ingest(connector, root, str(tmp_path),
                                      dtypes={'ENTITY1': Dtype.int})
    assert pq.read_table(prepared.source['paths'][0]).to_pydict() == {
        'ENTITY1': [1, 2]
    }
    for mode in (ReadMode.STAGED, 'staged'):
        backend.mode = mode
        with pytest.raises(UnsupportedReadModeError):
            backend.prepare_ingest(connector, root, str(tmp_path / 'staged'))


@pytest.mark.parametrize('path', [
    '/Volumes/example_catalog/example_schema/example_volume/../source',
    '/Volumes/example_catalog/example_schema/example_volume//source',
    '/Volumes/other_catalog/example_schema/example_volume/source',
    '/Volumes/example_catalog/other_schema/example_volume/source',
    '/Volumes/example_catalog/example_schema/example_volume',
])
def test_volume_scope_is_rejected_before_opening_client(volume_source, tmp_path,
                                                        path):
    backend, _, connector, _ = volume_source
    backend._client = lambda connector: pytest.fail(
        'Invalid scope opened a client')
    with pytest.raises(ValueError):
        backend.prepare_ingest(connector, path, str(tmp_path))
    with pytest.raises(ValueError):
        backend.get_data_version(connector, path)


@pytest.mark.parametrize('method,args', [('get_columns', ()),
                                         ('get_sample', (1, )),
                                         ('get_table_config', ())])
def test_volume_introspection_has_explicit_supported_scope(
        volume_source, method, args):
    backend, _, connector, path = volume_source
    backend._client = lambda connector: pytest.fail(
        'Introspection opened a client')
    with pytest.raises(PermanentSourceError,
                       match='training tables and custom sources only'):
        getattr(backend, method)(connector, path, *args)


@pytest.mark.parametrize('chunked', [False, True])
def test_same_size_volume_rewrite_during_read_is_rejected(
        volume_source, tmp_path, monkeypatch, chunked):
    from kumo_core.sources.errors import TransientSourceError
    from kumo_databricks.backend import _VolumeReader

    versions = []
    original_version = _VolumeReader.data_version

    def version(reader, table, **kwargs):
        versions.append(table)
        return original_version(reader, table, **kwargs)

    monkeypatch.setattr(_VolumeReader, 'data_version', version)
    backend, volume, connector, root = volume_source
    path = root + '.parquet'
    body = parquet([1, 2])
    physical = path
    if chunked:
        generation = path + '.chunks/generation/'
        physical = generation + '00000'
        second = generation + '00001'
        volume.objects.update({
            physical: body[:len(body) // 2],
            second: body[len(body) // 2:],
            generation + '_SUCCESS': b''
        })
        volume.modified.update({physical: 1000, second: 5000})
    else:
        volume.objects[path] = body
    original = volume.do

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        volume.modified[physical] = 2000 if chunked else 1_700_000_000_001
        return result

    volume.do = mutate
    with pytest.raises(TransientSourceError, match='changed while'):
        backend.prepare_ingest(connector, path, str(tmp_path))
    assert versions == [path, path]
    assert not list(tmp_path.rglob('*.parquet'))


@pytest.mark.parametrize('directory', [False, True])
def test_logical_volume_version_includes_every_physical_part(
        volume_source, directory):
    backend, volume, connector, root = volume_source
    path = root + '/part.parquet' if directory else root + '.parquet'
    body = parquet([1, 2, 3])
    generation = path + '.chunks/generation/'
    first, second = generation + '00000', generation + '00001'
    volume.objects.update({
        first: body[:len(body) // 2],
        second: body[len(body) // 2:],
        generation + '_SUCCESS': b''
    })
    volume.modified.update({first: 1000, second: 5000})
    selected = root if directory else path
    before = backend.get_data_version(connector, selected)
    volume.modified[first] = 2000
    assert backend.get_data_version(connector, selected) != before


def test_volume_parts_are_normalized_to_one_schema(volume_source, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    backend, volume, connector, root = volume_source
    volume.objects[root + '/a.parquet'] = parquet([1], pa.int32())
    volume.objects[root + '/b.parquet'] = parquet([2], pa.int64())
    result = backend.prepare_ingest(connector, root, str(tmp_path))
    tables = [pq.read_table(path) for path in result.source['paths']]
    assert len({str(table.schema) for table in tables}) == 1
    assert sorted(pa.concat_tables(tables)['ENTITY1'].to_pylist()) == [1, 2]


def test_volume_shards_must_each_have_declared_columns(volume_source, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumoapi.typing import Dtype
    backend, volume, connector, root = volume_source
    volume.objects[root + '/a.parquet'] = parquet([1])
    output = pa.BufferOutputStream()
    pq.write_table(pa.table({'SPLIT': [0]}), output)
    volume.objects[root + '/b.parquet'] = output.getvalue().to_pybytes()
    with pytest.raises(PermanentSourceError,
                       match='missing declared columns') as caught:
        backend.prepare_ingest(connector, root, str(tmp_path),
                               dtypes={'ENTITY1': Dtype.int})
    assert caught.value.table == root


@pytest.mark.parametrize('layout', ['nested', 'delta'])
def test_volume_rejects_ambiguous_or_transactional_layouts(
        volume_source, tmp_path, layout):
    backend, volume, connector, root = volume_source
    if layout == 'nested':
        volume.objects[root + '/run1/part.parquet'] = parquet([1])
        volume.objects[root + '/run2/part.parquet'] = parquet([2])
    else:
        volume.objects[root + '/part.parquet'] = parquet([1])
        volume.objects[root + '/_delta_log/000.json'] = b'{}'
    with pytest.raises(PermanentSourceError, match='Hive|Delta'):
        backend.prepare_ingest(connector, root, str(tmp_path))


def test_volume_source_location_and_errors_keep_the_scoped_path(
        volume_source, tmp_path):
    from kumo_core.sources.errors import SourceNotFoundError
    from kumoapi.typing import Dtype
    backend, volume, connector, root = volume_source
    volume.objects[root + '.parquet'] = parquet([1])
    path = root + '.parquet'
    assert backend.table_location(connector, path) == path
    with pytest.raises(SourceNotFoundError) as caught:
        backend.prepare_ingest(connector, path, str(tmp_path),
                               dtypes={'TARGET1': Dtype.int})
    assert caught.value.table == path and path in str(caught.value)


def test_volume_metadata_requests_do_not_grow_with_sibling_pairs(
        volume_source, tmp_path):
    backend, volume, connector, root = volume_source
    count = 64
    volume.objects.update(
        {root + f'/part-{i}.parquet': parquet([i])
         for i in range(count)})
    original = volume.list_directory_contents
    calls = []

    def listing(path):
        calls.append(path)
        return original(path)

    volume.list_directory_contents = listing
    backend.prepare_ingest(connector, root, str(tmp_path))
    assert len(calls) < count * 2
    calls.clear()
    reader = backend._volume_reader(connector, root)
    reader._identities([SimpleNamespace(path=path) for path in volume.objects],
                       None)
    assert calls == [root]


def test_volume_version_errors_use_source_error_classes(volume_source):
    from databricks.sdk.errors import PermissionDenied
    from kumo_core.sources.errors import SourceAccessDeniedError
    backend, volume, connector, root = volume_source

    def denied(path):
        raise PermissionDenied('metadata denied')

    volume.get_metadata = denied
    with pytest.raises(SourceAccessDeniedError) as caught:
        backend.get_data_version(connector, root + '.parquet')
    assert caught.value.table == root + '.parquet'


def test_volume_disappearance_during_schema_read_is_transient(
        volume_source, tmp_path):
    from kumo_core.sources.errors import TransientSourceError
    backend, volume, connector, root = volume_source
    path = root + '.parquet'
    volume.objects[path] = parquet([1, 2])
    original = volume.do

    def disappearing(*args, **kwargs):
        response = original(*args, **kwargs)
        volume.objects.pop(path, None)
        return response

    volume.do = disappearing
    with pytest.raises(TransientSourceError, match='changed while'):
        backend.prepare_ingest(connector, path, str(tmp_path))
    assert not list(tmp_path.rglob('*.parquet'))


def test_volume_hive_partition_folders_remain_supported(volume_source, tmp_path):
    import pyarrow.parquet as pq
    backend, volume, connector, root = volume_source
    volume.objects[root + '/day=2026/part.parquet'] = parquet([1, 2])
    prepared = backend.prepare_ingest(connector, root, str(tmp_path))
    frame = pq.read_table(prepared.source['paths'][0]).to_pandas()
    assert frame['ENTITY1'].tolist() == [1, 2]
    assert frame['day'].tolist() == [2026, 2026]


def test_extensionless_volume_parquet_remains_readable(volume_source, tmp_path):
    import pyarrow.parquet as pq
    backend, volume, connector, root = volume_source
    volume.objects[root] = parquet([1, 2])
    version = backend.get_data_version(connector, root)
    prepared = backend.prepare_ingest(connector, root, str(tmp_path))
    assert prepared.version == version
    assert pq.read_table(prepared.source['paths'][0])['ENTITY1'].to_pylist() == [1, 2]


@pytest.mark.parametrize('operation', ['version', 'ingest'])
def test_missing_volume_source_keeps_its_table_path(volume_source, tmp_path, operation):
    from kumo_core.sources.errors import SourceNotFoundError
    backend, volume, connector, root = volume_source
    with pytest.raises(SourceNotFoundError) as caught:
        if operation == 'version':
            backend.get_data_version(connector, root)
        else:
            backend.prepare_ingest(connector, root, str(tmp_path))
    assert caught.value.table == root
    assert root in str(caught.value)


@pytest.mark.parametrize('operation', ['version', 'ingest'])
def test_volume_part_deleted_during_first_version_is_transient(
        volume_source, tmp_path, monkeypatch, operation):
    from kumo_core.sources.errors import TransientSourceError
    from kumo_databricks.volume import VolumeSourceHandler
    backend, volume, connector, root = volume_source
    part = root + '/part.parquet'
    volume.objects[part] = parquet([1, 2])
    original = VolumeSourceHandler.object_identities

    def disappearing(handler, paths):
        volume.objects.pop(part, None)
        return original(handler, paths)

    monkeypatch.setattr(VolumeSourceHandler, 'object_identities', disappearing)
    with pytest.raises(TransientSourceError) as caught:
        if operation == 'version':
            backend.get_data_version(connector, root)
        else:
            backend.prepare_ingest(connector, root, str(tmp_path))
    assert caught.value.table == root
    assert root in str(caught.value)
    assert not list(tmp_path.rglob('*.parquet'))
