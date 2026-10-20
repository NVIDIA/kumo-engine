# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from diskgraph import format_versions
from kumoapi.graph import Edge
from kumoapi.typing import Stype

import kumo.engine._provenance as provenance
from kumo.engine import (
    FileConnector,
    Graph,
    Table,
    __version__,
    diff_manifests,
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


def test_fresh_snapshot_writes_manifest(make_snapshot, tmp_path):
    snapshot = make_snapshot(30)
    manifest = snapshot.manifest
    graph_id, short_version = snapshot.id.split('@')

    assert manifest['kind'] == 'graph_snapshot'
    assert manifest['schema_version'] == 1
    assert manifest['id'] == snapshot.id
    assert manifest['graph_id'] == graph_id
    assert manifest['data_version'][:16] == short_version
    assert len(manifest['data_version']) == 64
    assert snapshot.index_path.name == manifest['data_version']
    assert manifest['formats'] == format_versions()
    sources = manifest['parents']['source_tables']
    assert {
        alias: entry['source']
        for alias, entry in sources.items()
    } == {
        'ents': 'LOCAL_FILE:ents.parquet',
        'events': 'LOCAL_FILE:events.parquet',
    }
    assert all(entry['version'] for entry in sources.values())
    assert manifest['tables'] == {'ents': {'rows': 5}, 'events': {'rows': 30}}
    assert manifest['edges']['events_ENT_ID_to_ents_ID'] == 30
    assert set(manifest['environment']) == {
        'kumo_engine', 'build_commit', 'kumo_core', 'torch', 'python'
    }
    assert manifest['environment']['kumo_engine'] == __version__
    assert manifest['scheduler'] is None
    assert datetime.fromisoformat(manifest['created_at']).tzinfo is not None
    assert str(tmp_path) not in json.dumps(manifest)


def test_index_without_manifest_is_reused_and_backfilled(
        make_snapshot, monkeypatch):
    import kumo_core.diskgraph.ingest as ingest

    snapshot = make_snapshot(30)
    original = snapshot.manifest
    (snapshot.index_path / SNAPSHOT_MANIFEST_FILE).unlink()
    monkeypatch.setattr(ingest, 'execute_diskgraph_ingest',
                        lambda *args: pytest.fail('cached index re-ingested'))

    reused = make_snapshot()
    rebuilt = reused.manifest

    assert reused == snapshot
    assert diff_manifests(original, rebuilt) == {
        'created_at': (original['created_at'], None),
        'reconstructed': (None, True),
        **{
            f'{group}.{key}': (value, None)
            for group in ('environment', 'formats')
            for key, value in original[group].items()
        },
    }


def test_manifest_without_ingest_stats_keeps_identity(tmp_path, no_scheduler):
    from types import SimpleNamespace

    from kumo.engine.graph import _write_snapshot_manifest

    result = SimpleNamespace(index_path=str(tmp_path),
                             ingest_stats_path=str(tmp_path / 'missing.json'))
    sources = {'users': {'source': 'LOCAL_FILE:users.parquet', 'version': '1'}}
    _write_snapshot_manifest(result, 'g@d', 'g', 'd' * 64, sources, fresh=True)

    manifest = GraphSnapshot('g@d', tmp_path, {}).manifest
    assert manifest['id'] == 'g@d'
    assert manifest['parents']['source_tables'] == sources
    assert manifest['tables'] == {'users': {'rows': None}}
    assert manifest['edges'] == {}
    assert manifest['environment'] is not None


def test_manifest_write_failure_leaves_snapshot_unchanged(
        make_snapshot, monkeypatch):
    def fail(path, manifest):
        raise OSError('disk full')

    monkeypatch.setattr('kumo.engine.graph.write_manifest', fail)
    with pytest.warns(UserWarning, match='disk full'):
        snapshot = make_snapshot(30)

    assert snapshot.manifest is None
    assert (snapshot.index_path / '_SUCCESS').is_file()
    assert snapshot.index_path.name.startswith(snapshot.id.split('@')[1])


def test_appended_rows_show_in_manifest_diff(make_snapshot):
    before = make_snapshot(30).manifest
    after = make_snapshot(40).manifest

    diff = diff_manifests(before, after)

    assert diff['data_version'] == (before['data_version'],
                                    after['data_version'])
    version = 'parents.source_tables.events.version'
    assert version in diff and diff[version][0] != diff[version][1]
    assert diff['tables.events.rows'] == (30, 40)
    assert 'parents.source_tables.ents.version' not in diff
    assert 'tables.ents.rows' not in diff


def test_diff_manifests_reports_added_removed_and_whole_lists():
    a = {'kind': 'k', 'x': {'y': 1, 'gone': 2}, 'items': [1, 2]}
    b = {'kind': 'k', 'x': {'y': 1, 'new': 3}, 'items': [1, 3]}

    assert diff_manifests(a, b) == {
        'items': ([1, 2], [1, 3]),
        'x.gone': (2, None),
        'x.new': (None, 3),
    }
    assert diff_manifests({'kind': 'graph_snapshot'},
                          {'kind': 'model_bundle'}) == {
                              'kind': ('graph_snapshot', 'model_bundle')
                          }


def test_build_commit_comes_from_the_local_version_segment():
    assert provenance._build_commit('1.2.0+abc1234') == 'abc1234'


@pytest.mark.parametrize('env, processing, expected', [
    ({}, None, None),
    ({
        'KUMO_SCHEDULER': 'databricks',
        'KUMO_JOB_ID': '11',
        'KUMO_JOB_RUN_ID': '22',
        'TRAINING_JOB_NAME': 'ignored',
    }, None, ('databricks', '11', '22', 'env')),
    ({
        'KUMO_JOB_RUN_ID': 'run-1',
        'DATABRICKS_RUNTIME_VERSION': '16.4',
    }, None, ('databricks', None, 'run-1', 'env')),
    ({
        'KUMO_JOB_ID': 'job-1'
    }, None, ('other', 'job-1', None, 'env')),
    ({
        'KUMO_JOB_RUN_ID': 'run-9',
        'TRAINING_JOB_NAME': 'train-1',
        'TRAINING_JOB_ARN': 'arn:train-1',
    }, None, ('sagemaker', 'train-1', 'run-9', 'env')),
    ({
        'TRAINING_JOB_NAME': 'train-1',
        'TRAINING_JOB_ARN': 'arn:train-1',
    }, None, ('sagemaker', 'train-1', 'arn:train-1', 'auto')),
    ({}, {
        'ProcessingJobName': 'proc-1',
        'ProcessingJobArn': 'arn:proc-1',
    }, ('sagemaker', 'proc-1', 'arn:proc-1', 'auto')),
    ({
        'SNOWFLAKE_SERVICE_NAME': 'SCORE_JOB_A1B2C3D4'
    }, None, ('snowflake', None, 'SCORE_JOB_A1B2C3D4', 'auto')),
    ({
        'DATABRICKS_RUNTIME_VERSION': '16.4'
    }, None, ('databricks', None, None, 'auto')),
])
def test_scheduler_context(no_scheduler, env, processing, expected):
    for name, value in env.items():
        no_scheduler.setenv(name, value)
    if processing is not None:
        provenance.PROCESSING_JOB_CONFIG.write_text(json.dumps(processing))

    context = provenance.scheduler_context()

    if expected is None:
        assert context is None
    else:
        assert context == dict(
            zip(('platform', 'job_id', 'job_run_id', 'source'), expected))


def test_manifest_from_a_newer_schema_is_ignored(tmp_path):
    snapshot = GraphSnapshot('snapshot', tmp_path, {})
    assert snapshot.manifest is None

    (tmp_path / SNAPSHOT_MANIFEST_FILE).write_text(
        json.dumps({
            'kind': 'graph_snapshot',
            'schema_version': 2
        }))
    with pytest.warns(UserWarning, match='schema_version 2'):
        assert snapshot.manifest is None

    (tmp_path / SNAPSHOT_MANIFEST_FILE).write_text('{"kind": ')
    with pytest.warns(UserWarning, match='unreadable'):
        assert snapshot.manifest is None
