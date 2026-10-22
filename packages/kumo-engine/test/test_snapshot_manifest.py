# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
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
from kumo.engine.connector.identity import (
    DatabricksIdentity,
    SnowflakeIdentity,
)
from kumo.engine.graph import SNAPSHOT_MANIFEST_FILE, GraphSnapshot

_SCHEDULER_ENV = ('KUMO_SCHEDULER', 'KUMO_JOB_ID', 'KUMO_JOB_RUN_ID',
                  'TRAINING_JOB_NAME', 'TRAINING_JOB_ARN',
                  'SNOWFLAKE_SERVICE_NAME', 'DATABRICKS_RUNTIME_VERSION')


@pytest.fixture
def no_scheduler(tmp_path, monkeypatch):
    for name in _SCHEDULER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(provenance, 'PROCESSING_JOB_CONFIG',
                        tmp_path / 'processingjobconfig.json')
    return monkeypatch


@pytest.fixture
def make_snapshot(tmp_path, no_scheduler):
    no_scheduler.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
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

    assert manifest['kind'] == 'graph_snapshot'
    assert manifest['id'] == before.id
    data = (tmp_path / 'data').resolve()
    assert {a: e['location']
            for a, e in manifest['parents']['source_tables'].items()} == {
                'ents': str(data / 'ents.parquet'),
                'events': str(data / 'events.parquet'),
            }
    assert {e['version_kind']
            for e in manifest['parents']['source_tables'].values()
            } == {'timestamp'}
    assert 'tables' not in manifest and 'edges' not in manifest
    assert before.counts['rows'] == {'ents': 5, 'events': 30}
    assert before.counts['edges']['events_ENT_ID_to_ents_ID'] == 30
    assert manifest['environment']['kumo_engine'] == get_kumo_version()
    assert manifest['environment']['diskgraph']
    assert manifest['scheduler'] is None
    assert datetime.fromisoformat(manifest['created_at']).tzinfo is not None
    assert str(tmp_path / 'scratch') not in json.dumps(manifest)

    after = make_snapshot(40)
    assert after.counts['rows']['events'] == 40
    sources = manifest['parents']['source_tables']
    new_sources = after.manifest['parents']['source_tables']
    assert new_sources['ents'] == sources['ents']
    assert new_sources['events']['version'] != sources['events']['version']


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
    assert reused.manifest == {
        **original, 'created_at': None, 'environment': None,
        'scheduler': None, 'reconstructed': True
    }


def test_manifest_write_failure_leaves_snapshot_unchanged(
        make_snapshot, monkeypatch):
    def fail(path, manifest):
        raise OSError('disk full')

    monkeypatch.setattr('kumo.engine.graph.write_manifest', fail)
    with pytest.warns(UserWarning, match='disk full'):
        snapshot = make_snapshot(30)

    assert snapshot.manifest is None
    assert (snapshot.index_path / '_SUCCESS').is_file()


@pytest.mark.parametrize('env, processing, expected', [
    ({}, None, None),
    ({'KUMO_SCHEDULER': 'databricks', 'KUMO_JOB_ID': '11',
      'KUMO_JOB_RUN_ID': '22', 'TRAINING_JOB_NAME': 'ignored'}, None,
     ('databricks', '11', '22')),
    ({'KUMO_JOB_RUN_ID': 'run-9', 'TRAINING_JOB_NAME': 'train-1'}, None,
     ('sagemaker', 'train-1', 'run-9')),
    ({}, {'ProcessingJobName': 'proc-1', 'ProcessingJobArn': 'arn:proc-1'},
     ('sagemaker', 'proc-1', 'arn:proc-1')),
    ({'SNOWFLAKE_SERVICE_NAME': 'JOB_A1'}, None, ('snowflake', None, 'JOB_A1')),
])
def test_scheduler_context(no_scheduler, env, processing, expected):
    for name, value in env.items():
        no_scheduler.setenv(name, value)
    if processing is not None:
        provenance.PROCESSING_JOB_CONFIG.write_text(json.dumps(processing))

    context = provenance.scheduler_context()

    assert context == (expected and dict(
        zip(('platform', 'job_id', 'job_run_id'), expected)))


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
    assert entry == {'connector': None, 'location': None, 'version': '7',
                     'version_kind': 'unknown'}


@pytest.mark.parametrize('record_sources', [True, False])
def test_custom_sql_records_text_unless_sources_are_off(record_sources):
    version = SourceVersion('abc', 'hour_bucket', {'orders': '17'})
    entry = provenance.source_table(
        SnowflakeConnector(warehouse='wh', database='db', schema_name='sch'),
        'q', 'SELECT * FROM orders', version, record_sources=record_sources)
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
    assert entry.pop('normalized_sql_fingerprint') == reformatted[
        'normalized_sql_fingerprint'] == custom_sql_synthetic_name(
            'SELECT * FROM orders', ConnectorType.SNOWFLAKE)
    assert reformatted['location'] != entry['location']
    assert entry == expected


def test_record_sources_false_keeps_table_names_only(make_snapshot,
                                                     tmp_path):
    names = {'ents': 'ents', 'events': 'events.parquet'}
    redacted = make_snapshot(30, record_sources=False)
    sources = redacted.manifest['parents']['source_tables']
    assert {alias: e['location'] for alias, e in sources.items()} == names
    assert str(tmp_path) not in json.dumps(redacted.manifest)
    assert redacted.manifest['record_sources'] is False
    assert redacted.record_sources is False

    full = make_snapshot(record_sources=True)
    assert full == redacted
    assert full.record_sources is True
    assert full.manifest['parents']['source_tables']['ents'][
        'location'] == str((tmp_path / 'data' / 'ents.parquet').resolve())
    assert full.manifest['created_at'] == redacted.manifest['created_at']
