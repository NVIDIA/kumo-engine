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
from diskgraph import format_versions
from kumoapi.graph import Edge
from kumoapi.typing import Stype

import kumo.engine._provenance as provenance
from kumo.engine import FileConnector, Graph, S3Connector, Table, __version__
from kumo.engine._manifest import diff_manifests
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

    def make(events: int | None = None) -> GraphSnapshot:
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
                Table.from_source_table(connector['ents.parquet'],
                                        primary_key='ID').infer_metadata(),
                'events':
                Table.from_source_table(
                    connector['events.parquet'],
                    primary_key='EID').infer_metadata().with_column(
                        'ENT_ID', stype=Stype.ID),
            }, [Edge('events', 'ENT_ID', 'ents')]).snapshot()

    return make


def test_snapshot_manifest_records_sources_and_diffs(make_snapshot,
                                                     tmp_path):
    before = make_snapshot(30)
    manifest = before.manifest

    assert manifest['kind'] == 'graph_snapshot'
    assert manifest['id'] == before.id
    assert manifest['formats'] == format_versions()
    data = (tmp_path / 'data').resolve()
    assert {a: e['location']
            for a, e in manifest['parents']['source_tables'].items()} == {
                'ents': str(data / 'ents.parquet'),
                'events': str(data / 'events.parquet'),
            }
    assert manifest['tables'] == {'ents': {'rows': 5}, 'events': {'rows': 30}}
    assert manifest['edges']['events_ENT_ID_to_ents_ID'] == 30
    assert manifest['environment']['kumo_engine'] == __version__
    assert manifest['scheduler'] is None
    assert datetime.fromisoformat(manifest['created_at']).tzinfo is not None
    assert str(tmp_path / 'scratch') not in json.dumps(manifest)

    diff = diff_manifests(manifest, make_snapshot(40).manifest)
    assert diff['tables.events.rows'] == (30, 40)
    assert 'parents.source_tables.events.version' in diff
    assert 'parents.source_tables.ents.version' not in diff


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
    changed = diff_manifests(original, reused.manifest)
    assert changed['reconstructed'] == (None, True)
    assert all(key.split('.')[0] in ('created_at', 'reconstructed',
                                     'environment', 'formats')
               for key in changed)


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
    (SimpleNamespace(config=None, source_backend=None, database='DB',
                     schema_name='SCHEMA'), 'TT', None, 'DB.SCHEMA.TT'),
    (None, 'synthetic', 'SELECT 1', 'sql:' +
     hashlib.sha256(b'SELECT 1').hexdigest()),
])
def test_source_location(connector, name, sql, location):
    assert provenance.source_location(connector, name, sql) == location


def test_unrecordable_location_is_null():
    with pytest.warns(UserWarning, match='location'):
        entry = provenance.source_table(None, 'users', None, 7)
    assert entry == {'connector': None, 'location': None, 'version': '7'}
