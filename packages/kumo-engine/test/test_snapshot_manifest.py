# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from kumo_core.custom_sql import custom_sql_synthetic_name
from kumo_core.sources.identity import SourceVersion
from kumo_core.types import ConnectorType
from kumo_core.version import get_kumo_version
from kumoapi.graph import Edge
from kumoapi.typing import Stype

import kumo.engine._provenance as provenance
from kumo.engine import (
    BigQueryConnector,
    DatabricksConnector,
    FileConnector,
    Graph,
    S3Connector,
    SnowflakeConnector,
    Table,
)
from kumo.engine._manifest import (
    SnapshotManifest,
    SourceTableEntry,
    read_manifest,
    write_manifest,
)
from kumo.engine.connector.identity import (
    DatabricksIdentity,
    SnowflakeIdentity,
)
from kumo.engine.graph import SNAPSHOT_MANIFEST_FILE, GraphSnapshot


@pytest.fixture
def make_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    data = tmp_path / 'data'
    data.mkdir()
    pd.DataFrame({'ID': np.arange(5)}).to_parquet(data / 'ents.parquet')

    def make(events: int | None = None, **options) -> GraphSnapshot:
        if events is not None:
            pd.DataFrame({
                'EID': np.arange(events),
                'ENT_ID': np.arange(events) % 5,
            }).to_parquet(data / 'events.parquet')
            # Single-file versions are mtime seconds.
            stamp = 1.7e9 + events
            os.utime(data / 'events.parquet', (stamp, stamp))
        connector = FileConnector(root_dir=str(data))
        return Graph(
            {
                'ents':
                Table.from_source_table(connector['ents'],
                                        primary_key='ID').infer_metadata(),
                'events':
                Table.from_source_table(
                    connector['events.parquet'],
                    primary_key='EID').infer_metadata().with_column(
                        'ENT_ID', stype=Stype.ID),
            }, [Edge('events', 'ENT_ID', 'ents')]).snapshot(**options)

    return make


def test_snapshot_manifest_records_sources_and_diffs(make_snapshot,
                                                     tmp_path):
    before = make_snapshot(30)
    manifest = before.manifest

    assert manifest.kind == 'snapshot'
    assert SNAPSHOT_MANIFEST_FILE == '_provenance.json'
    assert manifest.id == before.id
    data = (tmp_path / 'data').resolve()
    assert {a: e.location
            for a, e in manifest.parents.source_tables.items()} == {
                'ents': str(data / 'ents.parquet'),
                'events': str(data / 'events.parquet'),
            }
    assert {e.version_kind
            for e in manifest.parents.source_tables.values()
            } == {'timestamp'}
    assert manifest.extra == {}
    assert manifest.environment.kumo_engine == get_kumo_version()
    assert manifest.environment.diskgraph
    assert 'scheduler' not in manifest.to_dict()
    assert datetime.fromisoformat(manifest.created_at).tzinfo is not None
    assert str(tmp_path / 'scratch') not in json.dumps(manifest.to_dict())

    after = make_snapshot(40)
    sources = manifest.parents.source_tables
    new_sources = after.manifest.parents.source_tables
    assert new_sources['ents'] == sources['ents']
    assert new_sources['events'].version != sources['events'].version


def test_a_snapshot_built_with_mlops_records_its_build(make_snapshot,
                                                       tmp_path):
    from kumo_core.diskgraph.stores import diskgraph_index_size_bytes

    from kumo.engine import MLOps

    plain = make_snapshot(10)
    built = make_snapshot(20, mlops=MLOps(artifact_root=str(tmp_path / 'a'),
                                          store={'model'}))

    assert plain.manifest.build is None
    build = built.manifest.build
    assert build.disk_bytes == diskgraph_index_size_bytes(built.index_path)
    assert build.started_at <= build.finished_at and build.seconds > 0


def test_index_without_manifest_is_reused_and_backfilled(
        make_snapshot, monkeypatch):
    import kumo_core.diskgraph.ingest as ingest

    snapshot = make_snapshot(30)
    original = snapshot.manifest
    (snapshot.index_path / SNAPSHOT_MANIFEST_FILE).unlink()
    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest',
                        lambda *args: pytest.fail('cached index re-ingested'))

    reused = make_snapshot()

    assert reused == snapshot
    assert reused.manifest == replace(original, created_at=None,
                                      environment=None,
                                      reconstructed=True)


def test_manifest_write_failure_leaves_snapshot_unchanged(
        make_snapshot, monkeypatch):
    def fail(path, manifest):
        raise OSError('disk full')

    monkeypatch.setattr('kumo.engine.graph.write_manifest', fail)
    with pytest.warns(UserWarning, match='disk full'):
        snapshot = make_snapshot(30)

    assert snapshot.manifest is None
    assert (snapshot.index_path / '_SUCCESS').is_file()


@pytest.mark.parametrize('content', [
    None, '{"schema_version": 2}', '{"schema_version": true}', '{"kind": '
])
def test_unsupported_manifest_is_ignored(tmp_path, content):
    if content is not None:
        (tmp_path / SNAPSHOT_MANIFEST_FILE).write_text(content)
        with pytest.warns(UserWarning):
            assert GraphSnapshot('s', tmp_path, {}).manifest is None
    else:
        assert GraphSnapshot('s', tmp_path, {}).manifest is None


@pytest.mark.parametrize('connector, name, sql, location', [
    (S3Connector(root_dir='s3://bucket-a/data/'), 'exports/tt/', None,
     's3://bucket-a/data/exports/tt'),
    (S3Connector(root_dir='s3://bucket-a/data/'), 'q4#1 50%.parquet', None,
     's3://bucket-a/data/q4%231%2050%25.parquet'),
    (S3Connector(), 's3://bucket-a/sales%23q4.parquet', None,
     's3://bucket-a/sales%23q4.parquet'),
    (SnowflakeConnector(warehouse='wh', database='db', schema_name='sch',
                        identity=SnowflakeIdentity(account='org-acct')),
     'TT', None, 'org-acct/"DB"."SCH"."TT"'),
    (SnowflakeConnector(warehouse='wh', database='db', schema_name='sch'),
     'TT', None, '"DB"."SCH"."TT"'),
    (DatabricksConnector(
        catalog='cat', schema_name='sch', warehouse_id='w',
        identity=DatabricksIdentity(host='https://workspace.example.com',
                                    client_id='c', client_secret='s')),
     'cat.sch.tt', None, '`cat`.`sch`.`tt`'),
    (BigQueryConnector(project_id='data-project', dataset_id='analytics'),
     'tt', None, 'data-project.analytics.tt'),
    (SimpleNamespace(source_type='OTHER', source_backend=None), 'TT', None,
     'TT'),
    (None, 'synthetic', 'SELECT 1', 'sql:' +
     hashlib.sha256(b'SELECT 1').hexdigest()),
])
def test_source_location(connector, name, sql, location):
    assert provenance.source_location(connector, name, sql) == location


def test_unrecordable_location_is_null():
    with pytest.warns(UserWarning, match='location'):
        entry = provenance.source_table(None, 'users', None, 7)
    assert entry == SourceTableEntry(connector=None, location=None,
                                     version='7', version_kind='unknown')


@pytest.mark.parametrize('record_sources', [True, False])
def test_custom_sql_records_text_unless_sources_are_off(record_sources):
    version = SourceVersion('abc', 'hour_bucket', {'orders': '17'})
    entry = provenance.source_table(
        SnowflakeConnector(warehouse='wh', database='db', schema_name='sch'),
        'q', 'SELECT * FROM orders', version,
        record_sources=record_sources).to_dict()
    expected = {
        'connector': 'SNOWFLAKE',
        'location': 'sql:' + hashlib.sha256(b'SELECT * FROM orders').hexdigest(),
        'version': 'abc',
        'version_kind': 'hour_bucket',
        'dependencies': {'orders': '17'},
    }
    if record_sources:
        expected.update(sql='SELECT * FROM orders', dialect='snowflake',
                        namespace=['db', 'sch'])
    reformatted = provenance.source_table(
        SnowflakeConnector(warehouse='wh', database='db', schema_name='sch'),
        'q', 'select *\n  from ORDERS -- note', version)
    assert entry.pop('normalized_sql_fingerprint') == (
        reformatted.normalized_sql_fingerprint) == custom_sql_synthetic_name(
            'SELECT * FROM orders', ConnectorType.SNOWFLAKE)
    assert reformatted.location != entry['location']
    assert entry == expected


def test_record_sources_false_keeps_table_names_only(make_snapshot,
                                                     tmp_path):
    names = {'ents': 'ents', 'events': 'events.parquet'}
    redacted = make_snapshot(30, record_sources=False)
    sources = redacted.manifest.parents.source_tables
    assert {alias: e.location for alias, e in sources.items()} == names
    assert str(tmp_path) not in json.dumps(redacted.manifest.to_dict())
    assert redacted.manifest.record_sources is False
    assert redacted.record_sources is False

    full = make_snapshot(record_sources=True)
    assert full == redacted
    assert full.record_sources is True
    assert full.manifest.parents.source_tables['ents'].location == str(
        (tmp_path / 'data' / 'ents.parquet').resolve())
    assert full.manifest.created_at == redacted.manifest.created_at


def test_toggling_record_sources_keeps_unknown_parent_fields(make_snapshot):
    written = make_snapshot(30).index_path / SNAPSHOT_MANIFEST_FILE
    data = json.loads(written.read_text())
    data['parents']['future'] = {'kept': True}
    written.write_text(json.dumps(data))

    redacted = make_snapshot(record_sources=False)
    assert redacted.manifest.record_sources is False
    assert redacted.manifest.parents.extra == {'future': {'kept': True}}


def test_snapshot_manifest_round_trips_unchanged(make_snapshot, tmp_path):
    written = make_snapshot(30).index_path / SNAPSHOT_MANIFEST_FILE
    newer = json.loads(written.read_text())
    newer['future'] = {'kept': [1, 2]}
    newer['environment']['cuda'] = '13.0'
    newer['parents']['source_tables']['ents']['row_count'] = 5
    source = tmp_path / 'newer.json'
    source.write_text(json.dumps(newer, indent=2, sort_keys=True) + '\n')

    manifest = read_manifest(source, SnapshotManifest)
    write_manifest(tmp_path / 'out.json', manifest)

    assert (tmp_path / 'out.json').read_bytes() == source.read_bytes()
    assert manifest.extra == {'future': {'kept': [1, 2]}}
    assert manifest.environment.extra == {'cuda': '13.0'}
