# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline Snowflake export-path regressions; no Temporal/REST server needed."""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from kumo_core.types import PredictionArtifactType

from io_utils import fsutil
from kumo.connector.connector import ConnectorError
from kumo.connector.snowflake_connector import (
    MAX_COPY_FILES,
    SnowflakeConnector,
)
from rest.backends.ml_backend.activities import parquet_io_activities
from rest.config.artifact_export import (
    ArtifactExportInput,
    ArtifactOutTable,
    NamedOutputConfig,
)
from rest.config.batch_prediction_job import OutputArtifactStats
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.jobs import WriteMode
from statedb.config.common import ConnectorID, ProjectID


def _connector(monkeypatch: pytest.MonkeyPatch) -> tuple:
    connector = SnowflakeConnector('output', account='acct', warehouse='wh',
                                   database='db', schema_name='sch')
    statements: list = []
    connector._context = Mock()
    monkeypatch.setattr(type(connector), 'context',
                        property(lambda self: self._context))
    monkeypatch.setattr(
        connector, 'execute_meta_query',
        lambda _ctx, query, creds_clause=None: statements.append(query))
    monkeypatch.setattr(connector, 'table_exists', lambda _name: True)
    return connector, statements


def _copy_statement(statements: list) -> str:
    copies = [item for item in statements if item.startswith('COPY INTO')]
    assert len(copies) == 1, statements
    return copies[0]


@pytest.mark.parametrize('multiple_roots', [False, True])
def test_snowflake_copies_every_selected_shard_not_only_the_first_root(
    monkeypatch: pytest.MonkeyPatch,
    multiple_roots: bool,
) -> None:
    """Every selected shard must reach COPY INTO, across attempt roots.

    Selection resolution, activity dispatch, export handling and the
    connector stay real; only manifest storage and Snowflake I/O are faked.
    """
    artifact_type = PredictionArtifactType.PREDICTIONS
    base = 's3://artifacts/job/_attempts'
    root = f'{base}/runtime-00000/att-b'
    other_root = f'{base}/runtime-00001/att-c'
    first = f'{root}/part-00000-00000.parquet'
    second = f'{other_root}/part-00001-00000.parquet'
    objects = {
        root: [first],
        other_root: [second],
        f'{base}/runtime-00000/att-a':
        [f'{base}/runtime-00000/att-a/p.parquet'],
    }
    roots = (root, other_root) if multiple_roots else (root, )
    expected = [first, second] if multiple_roots else [first]
    selected = SimpleNamespace(artifact_type=artifact_type, root=root,
                               roots=roots, file_count=len(expected))
    monkeypatch.setattr(
        parquet_io_activities, 'load_prediction_selection',
        lambda *_a, **_k: SimpleNamespace(artifacts=(selected, )))
    monkeypatch.setattr(fsutil, 'listdir', lambda path: objects[path])
    connector, statements = _connector(monkeypatch)
    monkeypatch.setattr(parquet_io_activities, 'get_connector_from_spec',
                        lambda _spec: connector)
    monkeypatch.setattr(
        'kumo.connector.snowflake_connector.get_temporary_aws_credential',
        lambda _path: {})

    output = NamedOutputConfig(
        connector=ConnectorExecutionRef(
            project_id=ProjectID('project_' + 'a' * 32),
            connector_id=ConnectorID('connector_' + 'b' * 32)),
        out_table=ArtifactOutTable(table_name='predictions',
                                   write_mode=WriteMode.APPEND),
    )
    input = ArtifactExportInput(
        artifact_paths={artifact_type: root},
        output_config={artifact_type: [output]},
        prediction_selection=Mock(),
        prediction_artifact_types=(artifact_type, ),
    )
    stats = OutputArtifactStats(artifact_type=artifact_type, num_rows=10,
                                num_cols=3, size_bytes=240, output_path=root)

    with patch.object(fsutil, 'copy',
                      side_effect=AssertionError('no local staging')):
        result = parquet_io_activities._write_artifacts_to_customer(
            input, artifact_stats={artifact_type: stats},
            resolved_connectors={artifact_type: [Mock()]})

    # The reported count is the selection aggregate; on master it was right
    # while only the first shard loaded, which is what made the bug silent.
    assert result[0].num_rows == 10
    assert result[0].output_path == 'DB.SCH.predictions'

    copy_sql = _copy_statement(statements)
    # The prefix descends as far as the selection allows: to the single
    # attempt directory, or to their shared parent when roots differ.
    prefix = f'{base}/' if multiple_roots else f'{root}/'
    assert f"FROM '{prefix}'" in copy_sql
    for path in expected:
        assert f"'{path[len(prefix):]}'" in copy_sql
    # The abandoned attempt sits under the same prefix and must not load.
    assert 'att-a' not in copy_sql
    assert 'MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE' in copy_sql


def test_snowflake_legacy_prefix_load_is_unchanged(
        monkeypatch: pytest.MonkeyPatch) -> None:
    connector, statements = _connector(monkeypatch)
    monkeypatch.setattr(
        'kumo.connector.snowflake_connector.get_temporary_aws_credential',
        lambda _path: {})
    connector.write_artifacts_to_table('OUT', 's3://artifacts/predictions')
    copy_sql = _copy_statement(statements)
    assert "FROM 's3://artifacts/predictions/'" in copy_sql
    assert 'FILES =' not in copy_sql


def test_snowflake_batches_selections_beyond_the_files_limit(
        monkeypatch: pytest.MonkeyPatch) -> None:
    connector, statements = _connector(monkeypatch)
    monkeypatch.setattr(
        'kumo.connector.snowflake_connector.get_temporary_aws_credential',
        lambda _path: {})
    # Snowflake's documented FILES cap, written out so that changing the
    # constant to a value Snowflake would reject fails this test.
    cap = 1000
    assert MAX_COPY_FILES == cap
    files = [
        f's3://artifacts/p/part-{index:05d}.parquet' for index in range(cap + 1)
    ]
    connector.write_artifacts_to_table('OUT', 's3://artifacts/p',
                                       artifact_files=files)
    copies = [item for item in statements if item.startswith('COPY INTO')]
    assert len(copies) == 2
    assert copies[0].count('.parquet') == cap
    assert copies[1].count('.parquet') == 1
    # Every selected object is named exactly once across the batches.
    loaded = sum(item.count('.parquet') for item in copies)
    assert loaded == len(files)


def test_snowflake_common_prefix_does_not_split_a_path_segment(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``runtime-00001`` and ``runtime-00010`` share characters, not a dir."""
    connector, statements = _connector(monkeypatch)
    monkeypatch.setattr(
        'kumo.connector.snowflake_connector.get_temporary_aws_credential',
        lambda _path: {})
    files = [
        's3://bucket/base/runtime-00001/part.parquet',
        's3://bucket/base/runtime-00010/part.parquet',
    ]
    connector.write_artifacts_to_table('OUT', 's3://bucket/base',
                                       artifact_files=files)
    copy_sql = _copy_statement(statements)
    assert "FROM 's3://bucket/base/'" in copy_sql
    assert "'runtime-00001/part.parquet'" in copy_sql
    assert "'runtime-00010/part.parquet'" in copy_sql


@pytest.mark.parametrize('files', [
    [],
    ['s3://b/p/a.parquet', 's3://b/p/a.parquet'],
    ['s3://b/p/'],
    ['s3://b/p/*.parquet'],
    ['gs://b/p/a.parquet'],
])
def test_snowflake_rejects_unusable_selections_before_any_io(
        monkeypatch: pytest.MonkeyPatch, files: list) -> None:
    connector, statements = _connector(monkeypatch)
    with pytest.raises(ConnectorError, match='Selected Snowflake artifacts'):
        connector.write_artifacts_to_table('OUT', 's3://b/p',
                                           artifact_files=files)
    assert statements == []


def test_snowflake_local_staging_uploads_only_selected_shards(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The real upload body must stage the selection, not rescan the root.

    Stubbing the upload helper would prove only argument forwarding, so this
    fakes the cursor underneath it and observes what is actually staged.
    """
    connector, statements = _connector(monkeypatch)
    selected = tmp_path / 'att-b' / 'part-00000.parquet'
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b'x')
    abandoned = tmp_path / 'att-a' / 'part-00000.parquet'
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b'y')

    uploaded: list = []

    class _Cursor:
        def _upload(self, local_file_name, stage_location, options):
            uploaded.append(local_file_name)

        def close(self):
            pass

    connector._context.cursor = _Cursor
    connector.write_artifacts_to_table('OUT', str(tmp_path),
                                       artifact_files=[str(selected)])

    # The abandoned sibling sits on disk under the same root, so a body that
    # rescanned the root instead of honouring the selection would stage it.
    assert len(uploaded) == 1
    assert 'att-b' in uploaded[0]
    assert 'att-a' not in uploaded[0]
    copy_sql = _copy_statement(statements)
    # The isolated stage prefix now holds exactly the selection.
    assert 'FILES =' not in copy_sql
    assert 'PURGE = TRUE' in copy_sql


def test_snowflake_local_selection_rejects_one_file_named_twice(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Two spellings of one file must not be staged and loaded twice.

    The caller's uniqueness check compares the strings it was given, so
    'part.parquet' and './part.parquet' both survive it and collapse only
    once resolved. Uploading the survivor twice would put it under two stage
    prefixes and double its rows in the customer table.
    """
    connector, statements = _connector(monkeypatch)
    shard = tmp_path / 'att-a' / 'part-00000.parquet'
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b'x')
    # os.path.join, not pathlib: pathlib collapses the '.' segment and
    # would hand back a string identical to the first, which the
    # caller's string-level check already rejects.
    aliased = os.path.join(str(shard.parent), '.', shard.name)

    uploaded: list = []

    class _Cursor:
        def _upload(self, local_file_name, stage_location, options):
            uploaded.append(local_file_name)

        def close(self):
            pass

    connector._context.cursor = _Cursor
    with pytest.raises(ValueError, match='same file twice'):
        connector.write_artifacts_to_table('OUT', str(tmp_path),
                                           artifact_files=[str(shard), aliased])

    # Refused before anything is staged or executed.
    assert uploaded == []
    assert statements == []


def test_snowflake_refuses_a_widening_prefix_before_dropping_the_table(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused selection must not destroy the table it was replacing.

    ``_common_prefix`` rejects a selection that agrees only at the bucket.
    Computing it after the DROP would leave the customer's table deleted with
    nothing loaded in its place, so the refusal has to come first.
    """
    connector, statements = _connector(monkeypatch)
    with pytest.raises(ConnectorError):
        connector.write_artifacts_to_table(
            'OUT', 's3://bucket', replace=True, artifact_files=[
                's3://bucket/shard1/a.parquet', 's3://bucket/shard2/b.parquet'
            ])
    assert statements == [], 'DROP TABLE ran before the selection was refused'


def test_snowflake_refuses_a_selection_for_stage_backed_paths(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage FROM clause is unquoted, so a selection cannot be spliced in.

    Stage-backed artifacts are not routed through a selection; refusing beats
    emitting an internal URI as Snowflake SQL.
    """
    connector, statements = _connector(monkeypatch)
    with pytest.raises(ConnectorError, match='stage-backed'):
        connector.write_artifacts_to_table(
            'OUT', '@DB.SCH.STAGE/job',
            artifact_files=['@DB.SCH.STAGE/job/part-00000.parquet'])
    assert statements == []


@pytest.mark.parametrize('hostile', [
    "s3://b/p/a'.parquet",
    's3://b/p/a";DROP TABLE x.parquet',
    's3://b/p/a\\.parquet',
    's3://b/p/a\n.parquet',
    's3://b/p/a;.parquet',
])
def test_snowflake_rejects_sql_metacharacters_in_selected_paths(
        monkeypatch: pytest.MonkeyPatch, hostile: str) -> None:
    """Selected paths are interpolated into SQL, so they get the root's rules."""
    connector, statements = _connector(monkeypatch)
    with pytest.raises(ConnectorError):
        connector.write_artifacts_to_table(
            'OUT', 's3://b/p', artifact_files=['s3://b/p/ok.parquet', hostile])
    assert statements == []


def test_snowflake_local_selection_rejects_a_symlinked_shard(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Bypassing the root scan must not bypass its symlink rule."""
    connector, statements = _connector(monkeypatch)
    outside = tmp_path / 'outside.parquet'
    outside.write_bytes(b'x')
    link = tmp_path / 'root' / 'part-00000.parquet'
    link.parent.mkdir()
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        connector.write_artifacts_to_table('OUT', str(tmp_path / 'root'),
                                           artifact_files=[str(link)])
    assert statements == []


@pytest.mark.parametrize(('files', 'reason'), [
    (['s3://b/p/a.parquet', 's3://b/p2/b.parquet'], 'container root'),
    (['s3://b1/a.parquet', 's3://b2/b.parquet'], 'different buckets'),
])
def test_snowflake_refuses_a_prefix_that_widens_to_the_container(
        monkeypatch: pytest.MonkeyPatch, files: list, reason: str) -> None:
    """The shared prefix scopes the temp stage URL and the storage credential.

    A selection that only agrees at the bucket would silently widen both, so
    it is refused rather than loaded.
    """
    connector, statements = _connector(monkeypatch)
    with pytest.raises(ConnectorError):
        connector.write_artifacts_to_table('OUT', 's3://b/p',
                                           artifact_files=files)
    assert statements == [], reason


def test_snowflake_scheme_agreement_is_case_insensitive(
        monkeypatch: pytest.MonkeyPatch) -> None:
    connector, statements = _connector(monkeypatch)
    monkeypatch.setattr(
        'kumo.connector.snowflake_connector.get_temporary_aws_credential',
        lambda _path: {})
    connector.write_artifacts_to_table('OUT', 'S3://b/p/q',
                                       artifact_files=['S3://b/p/q/a.parquet'])
    assert "FROM 'S3://b/p/q/'" in _copy_statement(statements)


def test_snowflake_infers_the_new_table_schema_from_a_selected_object(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """First export creates the table, so inference must not read a stray file.

    Every other test stubs ``table_exists`` to True, which skips the whole
    create-table branch; this one exercises it.
    """
    connector, statements = _connector(monkeypatch)
    monkeypatch.setattr(connector, 'table_exists', lambda _name: False)
    monkeypatch.setattr(
        'kumo.connector.snowflake_connector.get_temporary_aws_credential',
        lambda _path: {})
    files = [
        's3://b/job/_attempts/r0/att-b/part-00000.parquet',
        's3://b/job/_attempts/r1/att-c/part-00001.parquet',
    ]
    connector.write_artifacts_to_table('OUT', 's3://b/job/_attempts',
                                       artifact_files=files)
    infer = [item for item in statements if 'INFER_SCHEMA' in item]
    assert len(infer) == 1
    # Naming one selected object keeps inference off any abandoned attempt
    # that happens to sit under the shared prefix.
    assert "FILES => ('r0/att-b/part-00000.parquet')" in infer[0]
    stage = [item for item in statements if 'CREATE TEMPORARY STAGE' in item]
    assert "URL = 's3://b/job/_attempts/'" in stage[0]
