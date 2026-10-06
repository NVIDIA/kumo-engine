# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise public filesystem ingest and its destructive-operation boundaries."""

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.artifact_export import ArtifactDestinationConflict
from kumo_core.sources import layout
from kumo_core.sources.errors import PermanentSourceError, SourceNotFoundError
from kumoapi.graph import Edge
from kumoapi.jobs import WriteMode
from kumoapi.source_table import LocalFileSourceTable
from kumoapi.table import TableDefinition as HostedTableDefinition
from kumoapi.typing import Dtype
from pydantic import TypeAdapter, ValidationError

from kumo.engine import FileConnector, Graph, Table, configure


@pytest.fixture
def local(tmp_path):
    root = tmp_path / 'data.with spaces'
    root.mkdir()
    (root / 'users.csv').write_text('id,name\n1,Alice\n2,Bob\n')
    pq.write_table(pa.table({
        'user_id': [1, 2],
        'amount': [2., 4.]
    }), root / 'events.parquet')
    configure(scratch_dir=str(tmp_path / 'scratch'))
    return FileConnector(root_dir=str(root)), root


def test_missing_root_names_local_directory(tmp_path):
    missing = tmp_path / 'absent'
    connector = FileConnector(root_dir=str(missing))
    with pytest.raises(SourceNotFoundError) as error:
        connector.table_names()
    message = str(error.value)
    assert str(missing) in message
    assert 'local directory' in message
    assert 'read and traverse' in message
    assert 'bucket' not in message and 'prefix' not in message


def test_public_graph_snapshot(local, tmp_path):
    connector, root = local
    original_mtime = (root / 'users.csv').stat().st_mtime
    assert connector.table_names() == ['events', 'users']
    users = Table.from_source_table(connector['users'], primary_key='id')
    events = Table.from_source_table(connector['events'])
    graph = Graph(
        tables=[users, events],
        edges=[Edge(src_table='events', fkey='user_id', dst_table='users')])
    graph.validate()
    snapshot = graph.snapshot()
    assert graph.snapshot().id == snapshot.id
    (root / 'users.csv').write_text('id,name\n1,Alice\n2,Bob\n3,Charlie\n')
    # Local source versions intentionally use whole-second modification
    # times. Model a later source update instead of depending on test runtime
    # to cross that boundary.
    os.utime(root / 'users.csv', (original_mtime + 2, original_mtime + 2))
    assert graph.snapshot().id != snapshot.id
    other = tmp_path / 'other.root'
    other.mkdir()
    (other / 'users.csv').write_text((root / 'users.csv').read_text())
    other_table = Table.from_source_table(
        FileConnector(root_dir=str(other))['users'], primary_key='id')
    assert Graph(tables=[users], edges=[]).id != Graph(tables=[other_table],
                                                       edges=[]).id
    assert len(
        graph._graph_config['tables'][0]['source_table_name'].split('.')) == 3


def test_hosted_schema_stays_closed(local):
    _, root = local
    with pytest.raises(ValidationError, match='union_tag_invalid'):
        TypeAdapter(HostedTableDefinition).validate_python({
            'cols': [],
            'source_table':
            LocalFileSourceTable(path=str(root / 'users.csv'))
        })


@pytest.mark.parametrize('name', [
    '', '../users', '/users', 'file:///users', 'x/../users', 'users?x',
    'users\\x'
])
def test_invalid_source_paths(local, name):
    connector, _ = local
    with pytest.raises(ValueError, match='relative path'):
        Table.from_source_table(connector[name])


def test_empty_source_and_unknown_column(local, tmp_path):
    connector, root = local
    (root / 'empty.csv').write_text('id,value\n')
    with pytest.raises(PermanentSourceError, match='(?i)(empty|no rows)'):
        connector.source_backend.prepare_ingest(connector, 'empty',
                                                str(tmp_path / 'empty-stage'))
    with pytest.raises(ValueError, match='(?i)(column|unknown)'):
        Table.from_source_table(connector['users'], column_names=['missing'])


def test_empty_parquet_table_names_the_diskgraph_row_requirement(
        local, tmp_path):
    connector, root = local
    pq.write_table(
        pa.table({
            'id': pa.array([], type=pa.int64()),
            'name': pa.array([], type=pa.string()),
            'amount': pa.array([], type=pa.float64()),
            'active': pa.array([], type=pa.bool_()),
        }), root / 'empty_table.parquet')
    with pytest.raises(PermanentSourceError, match='at least one row'):
        Table.from_source_table(connector['empty_table'])


def test_read_symlinks_and_overlap(local, tmp_path):
    connector, root = local
    outside = tmp_path / 'outside.csv'
    outside.write_text('id\n9\n')
    (root / 'escape.csv').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        Table.from_source_table(connector['escape'])
    (root / 'escape.csv').unlink()
    (root / 'alias.csv').symlink_to(root / 'users.csv')
    assert Table.from_source_table(connector['alias']).definition is not None
    (root / 'loop').symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        connector.table_names()
    (root / 'loop').unlink()
    table = Table.from_source_table(connector['users'], primary_key='id')
    before = (root / 'users.csv').read_bytes()
    configure(scratch_dir=str(root / 'scratch'))
    with pytest.raises(ValueError, match='overlap'):
        Graph(tables=[table], edges=[]).snapshot()
    assert (root / 'users.csv').read_bytes() == before
    assert not (root / 'scratch').exists()
    with pytest.raises(ValueError, match='overlap'):
        connector.source_backend.prepare_ingest(connector, 'users', str(root))


@pytest.mark.parametrize('name',
                         ['parts', 'parts/', 'parts.csv', 'parts.parquet'])
@pytest.mark.parametrize(
    'link_kind',
    ['internal_directory', 'external_directory', 'cycle', 'escaping_file'])
def test_unsafe_directory_rejected_before_scan(local, tmp_path, monkeypatch,
                                               name, link_kind):
    connector, root = local
    directory = root / name
    nested = directory / 'nested'
    nested.mkdir(parents=True)
    (directory / 'part.csv').write_text('id\n1\n')
    if link_kind == 'internal_directory':
        target = root / 'safe'
        target.mkdir()
    elif link_kind == 'external_directory':
        target = tmp_path / 'outside'
        target.mkdir()
    elif link_kind == 'cycle':
        target = directory
    else:
        target = tmp_path / 'outside.csv'
        target.write_text('id\n9\n')
    (nested / 'link').symlink_to(target, target_is_directory=target.is_dir())

    def unexpected_scan(*args, **kwargs):
        pytest.fail('Shared recursive scan reached an unsafe directory')

    monkeypatch.setattr(layout, 'scan', unexpected_scan)
    with pytest.raises(ValueError, match='symlinks'):
        connector.source_table_for(name)


@pytest.mark.parametrize('suffix', ['csv', 'parquet'])
def test_safe_directory_source(local, suffix):
    connector, root = local
    directory = root / 'parts'
    directory.mkdir()
    filename = 'users.csv' if suffix == 'csv' else 'events.parquet'
    (directory / filename).symlink_to(root / filename)
    source = connector.source_table_for('parts/')
    assert source.path == str(directory.resolve())
    assert source.file_type.value == suffix.upper()
    table = Table.from_source_table(connector['parts'])
    assert table.definition is not None
    assert connector.source_backend.get_data_version(connector, 'parts')


@pytest.mark.parametrize(
    'directory',
    ['outputs', 'space here', 'hash#data', 'query?data', 'percent%20data'])
def test_artifact_outputs(tmp_path, directory):
    connector = FileConnector(root_dir=str(tmp_path / directory))
    frame = pd.DataFrame({
        'id': [1, 2],
        'prediction': [0.2, 0.8],
        'embedding': [[1., 2.], [3., 4.]]
    })
    uri = connector.sink_backend.write_table(connector, 'predictions', frame)
    path = Path(unquote(urlsplit(uri).path))
    assert path.suffix == '.parquet'
    assert pq.read_table(path).to_pydict() == pa.Table.from_pandas(
        frame, preserve_index=False).to_pydict()
    csv_uri = connector.sink_backend.write_table(connector, 'embeddings.csv',
                                                 frame)
    assert pd.read_csv(csv_uri)['id'].tolist() == frame['id'].tolist()
    assert connector.source_backend.get_download_urls(
        connector, 'embeddings.csv') == [csv_uri]
    uri = connector.sink_backend.write_table(connector, 'parts/', frame)
    assert uri.endswith('/')
    with pytest.raises(ValueError, match='already contains'):
        connector.sink_backend.write_table(connector, 'parts/', frame)
    with pytest.raises(NotImplementedError):
        connector.sink_backend.write_table(connector, 'append', frame,
                                           WriteMode.APPEND)


@pytest.mark.parametrize('destination', ['models/model', 'models/model%20'])
def test_sink_escape_and_bundle_copy(local, tmp_path, destination):
    connector, root = local
    outside = tmp_path / 'outside.parquet'
    outside.write_bytes(b'preserve')
    (root / 'escape.parquet').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        connector.sink_backend.write_table(connector, 'escape',
                                           pd.DataFrame({'id': [1]}))
    assert outside.read_bytes() == b'preserve'
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'weights.bin').write_bytes(b'weights')
    published = connector.sink_backend.publish_model(connector, str(bundle),
                                                     path=destination)
    target = Path(unquote(urlsplit(published.uri).path)) / 'weights.bin'
    assert target.read_bytes() == b'weights'
    assert target.stat().st_ino != (bundle / 'weights.bin').stat().st_ino
    assert not target.is_symlink()
    with pytest.raises(ValueError, match='empty destination'):
        connector.sink_backend.publish_model(connector, str(bundle),
                                             path=destination)
    with pytest.raises(ValueError, match='existing nonempty'):
        connector.sink_backend.publish_model(connector,
                                             str(tmp_path / 'missing'),
                                             path='missing')
    output = FileConnector(root_dir=str(tmp_path))
    for destination in ('bundle', 'bundle/nested'):
        with pytest.raises(ValueError, match='overlap'):
            output.sink_backend.publish_model(output, str(bundle),
                                              path=destination)
    with pytest.raises(NotImplementedError):
        output.sink_backend.fetch_model(output, str(bundle), path='models')


def test_dot_names_metadata_stamp_and_permissions(local, tmp_path):
    connector, root = local
    dotted = root / 'users.v1.csv'
    dotted.write_text('id,name\n1,Alice\n')
    table = Table.from_source_table(connector['users.v1'], primary_key='id')
    Graph(tables=[table], edges=[]).validate()
    version = connector.source_backend.get_data_version(connector, 'users.v1')
    stat = dotted.stat()
    dotted.write_text('id,name\n1,Other\n')
    os.utime(dotted, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert connector.source_backend.get_data_version(connector,
                                                     'users.v1') == version
    destination = tmp_path / 'readonly'
    destination.mkdir()
    output = FileConnector(root_dir=str(destination))
    destination.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            output.sink_backend.write_table(output, 'denied',
                                            pd.DataFrame({'id': [1]}))
    finally:
        destination.chmod(0o755)


def test_directory_links_and_empty_bundle(local, tmp_path):
    connector, root = local
    (root / 'sub').mkdir()
    (root / 'alias').symlink_to(root / 'sub', target_is_directory=True)
    with pytest.raises(ValueError, match='Directory symlinks'):
        connector.sink_backend.write_table(connector, 'alias/output',
                                           pd.DataFrame({'id': [1]}))
    bundle = tmp_path / 'empty-bundle'
    (bundle / 'nested').mkdir(parents=True)
    with pytest.raises(ValueError, match='existing nonempty'):
        connector.sink_backend.publish_model(connector, str(bundle),
                                             path='models/empty')


@pytest.mark.parametrize(
    'name', ['', '..', '../escape', '/tmp/escape', 'file:///tmp/escape'])
def test_invalid_outputs(local, name):
    connector, _ = local
    with pytest.raises(ValueError, match='relative path'):
        connector.sink_backend.write_table(connector, name,
                                           pd.DataFrame({'id': [1]}))


def test_unsupported_suffix_and_conversion_error(local, tmp_path):
    connector, root = local
    (root / 'unsupported.json').write_text('{"id":1}')
    with pytest.raises(PermanentSourceError):
        Table.from_source_table(connector['unsupported.json'])
    (root / 'conversion.csv').write_text('id,value\n1,not-a-number\n')
    with pytest.raises(PermanentSourceError,
                       match='(?i)(convert|parse|int|numeric)'):
        connector.source_backend.prepare_ingest(
            connector, 'conversion', str(tmp_path / 'conversion-stage'), {
                'id': Dtype.int64,
                'value': Dtype.int64
            })


def test_hosted_table_definition_remains_usable_without_provider_io():
    from kumoapi.source_table import FileType, S3RelativeSource, S3SourceTable
    from kumoapi.table import Column
    from kumoapi.typing import Stype

    from kumo.engine import S3Connector

    connector = S3Connector(root_dir='s3://schema-only-fixture')
    source = S3SourceTable(connector_id=connector.id,
                           file_type=FileType.PARQUET,
                           source=S3RelativeSource(path='users'))
    definition = HostedTableDefinition(
        cols=[Column(name='id', dtype=Dtype.int64, stype=Stype.ID)], pkey='id',
        source_table=source)
    graph = Graph(tables=[Table(connector, 'users', definition)], edges=[])
    graph.validate()
    assert graph._graph_config['tables'][0]['source_table_name'].startswith(
        's3.')


@pytest.mark.parametrize("values", [
    ["999999-01-01"],
    [b"not-a-time"],
    [None],
])
def test_snapshot_rejects_invalid_required_times(local, values):
    from diskgraph import DiskgraphDataError
    from kumo_core.diskgraph.runner import NonRetryableDiskgraphError
    from kumoapi.typing import Stype
    connector, root = local
    timestamps = pa.array(
        values,
        type=pa.binary() if isinstance(values[0], bytes) else pa.string())
    pq.write_table(pa.table({
        "id": range(len(values)),
        "ts": timestamps
    }), root / "bad.parquet")
    table = Table.from_source_table(connector["bad"], time_column="ts")
    table = table.with_column("ts", dtype=Dtype.time, stype=Stype.timestamp)
    with pytest.raises((NonRetryableDiskgraphError, DiskgraphDataError),
                       match="(create.time|no usable rows)"):
        Graph(tables=[table], edges=[]).snapshot()
    assert not list((root.parent / "scratch").rglob("_SUCCESS"))


def test_snapshot_rejects_missing_time_in_later_shard(local):
    connector, root = local
    parts = root / "partial"
    parts.mkdir()
    pq.write_table(pa.table({
        "id": [1],
        "ts": ["2026-01-02T12:00:00"]
    }), parts / "part-0.parquet")
    pq.write_table(pa.table({"id": [2]}), parts / "part-1.parquet")
    table = Table.from_source_table(connector["partial"], time_column="ts")
    with pytest.raises(PermanentSourceError,
                       match="missing declared columns: ts"):
        Graph(tables=[table], edges=[]).snapshot()
    assert not list((root.parent / "scratch").rglob("_SUCCESS"))


@pytest.mark.parametrize("values", [
    [b"2026-01-02T12:00:00", b"2026-01-03T12:00:00"],
    [None, "2026-01-02T12:00:00"],
    ["not-a-time", "2026-01-02T12:00:00"],
    ["2026-01-02T12:00:00", "not-a-time"],
])
def test_snapshot_retains_only_usable_required_times(local, values):
    from kumoapi.typing import Stype
    connector, root = local
    pq.write_table(pa.table({
        "id": range(len(values)),
        "ts": values
    }), root / "valid.parquet")
    table = Table.from_source_table(connector["valid"], time_column="ts")
    table = table.with_column("ts", dtype=Dtype.time, stype=Stype.timestamp)
    snapshot = Graph(tables=[table], edges=[]).snapshot()
    assert (snapshot.index_path / "_SUCCESS").is_file()


def test_snapshot_filters_bad_time_in_later_shard(local):
    import json

    from kumoapi.typing import Stype
    connector, root = local
    parts = root / "filtered"
    parts.mkdir()
    for n, value in enumerate(["2026-01-02T12:00:00", "not-a-time"]):
        pq.write_table(pa.table({
            "id": [n],
            "ts": [value]
        }), parts / f"part-{n}.parquet")
    table = Table.from_source_table(connector["filtered"], time_column="ts")
    table = table.with_column("ts", dtype=Dtype.time, stype=Stype.timestamp)
    graph = Graph(tables=[table], edges=[])
    snapshot = graph.snapshot()
    stats = json.loads((snapshot.index_path / "ingest_stats.json").read_text())
    assert list(stats["stats"]["node_counts"].values()) == [1]
    assert list(stats["stats"]["dropped_rows_counts"].values()) == [1]
    assert graph.snapshot().id == snapshot.id  # Cache accepts discarded rows.


_NON_ROOT = pytest.mark.skipif(os.geteuid() == 0,
                               reason='root bypasses mode permissions')


@pytest.mark.parametrize('destination',
                         ['new/nested/result/', 'existing/', 'out', 'kept'])
def test_output_preflight_leaves_no_files(local, destination):
    connector, root = local
    (root / 'existing').mkdir()
    (root / 'kept.parquet').write_bytes(b'keep')
    before = set(root.rglob('*'))
    connector.sink_backend.preflight_write_files(connector, destination)
    assert set(root.rglob('*')) == before
    assert (root / 'kept.parquet').read_bytes() == b'keep'


@_NON_ROOT
@pytest.mark.parametrize('denied, destination', [
    ('.', 'result.parquet'),
    ('out.parquet', 'out'),
    ('res', 'res'),
    ('.', 'res'),
    ('locked', 'locked/new/results/'),
])
def test_output_preflight_rejects_unwritable_destination(
        local, denied, destination):
    connector, root = local
    (root / 'res').mkdir()
    (root / 'locked').mkdir()
    (root / 'out.parquet').write_bytes(b'keep')
    before = set(root.rglob('*'))
    (root / denied).chmod(0o500)
    try:
        with pytest.raises(PermissionError) as error:
            connector.sink_backend.preflight_write_files(connector, destination)
    finally:
        (root / denied).chmod(0o700)
    assert error.value.filename == f'{root}/{destination}'
    assert set(root.rglob('*')) == before
    assert (root / 'out.parquet').read_bytes() == b'keep'


@pytest.mark.parametrize('link', [False, True])
def test_output_preflight_rejects_non_directory_parent(local, link):
    connector, root = local
    if link:
        (root / 'results').symlink_to(root / 'missing-target')
    else:
        (root / 'results').write_bytes(b'keep')
    with pytest.raises(NotADirectoryError) as error:
        connector.sink_backend.preflight_write_files(connector, 'results/out/')
    assert error.value.filename == str(root / 'results' / 'out') + '/'
    assert not list(root.glob('.kumo-write-probe-*'))


@_NON_ROOT
def test_output_permissions_are_rechecked_at_publication(local):
    connector, root = local
    sink = connector.sink_backend
    sink.preflight_write_files(connector, 'results/')
    root.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            sink.write_files(connector, 'results/',
                             [str(root / 'events.parquet')])
    finally:
        root.chmod(0o700)
    assert not (root / 'results').exists()


@pytest.mark.parametrize('foreign', ['old.txt', '_KUMO_EXPORT'])
def test_output_preflight_rejects_occupied_directory(local, foreign):
    connector, root = local
    sink = connector.sink_backend
    (root / 'results').mkdir()
    (root / 'results' / foreign).write_text('other')
    before = set(root.rglob('*'))
    with pytest.raises(ArtifactDestinationConflict, match='another export'):
        sink.preflight_write_files(connector, 'results/')
    # Publication refuses the same destination, so the check cannot drift.
    with pytest.raises(ArtifactDestinationConflict):
        sink.write_files(connector, 'results/', [str(root / 'events.parquet')])
    assert set(root.rglob('*')) == before


def test_output_preflight_accepts_empty_directory(local):
    connector, root = local
    (root / 'results').mkdir()
    connector.sink_backend.preflight_write_files(connector, 'results/')
    assert list((root / 'results').iterdir()) == []


@_NON_ROOT
@pytest.mark.parametrize('destination', ['locked/out.parquet', 'locked/out'])
def test_output_preflight_overwrites_existing_file_in_readonly_directory(
        local, destination):
    connector, root = local
    sink = connector.sink_backend
    (root / 'locked').mkdir()
    (root / 'locked' / 'out.parquet').write_bytes(b'old')
    (root / 'locked').chmod(0o500)
    try:
        sink.preflight_write_files(connector, destination)
        # Publication rewrites the file in place, so preflight must agree.
        sink.write_files(connector, destination, [str(root / 'events.parquet')])
        assert (root / 'locked' / 'out.parquet').read_bytes() != b'old'
    finally:
        (root / 'locked').chmod(0o700)
