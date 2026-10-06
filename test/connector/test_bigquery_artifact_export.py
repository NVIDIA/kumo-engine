# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline BigQuery export-path regressions; no Temporal/REST server needed."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import google.cloud.bigquery as bigquery
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from google.api_core.exceptions import Forbidden, NotFound
from kumo_core.types import PredictionArtifactType

from io_utils import fsutil
from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.connector.connector import ConnectorError
from rest.backends.ml_backend.activities import parquet_io_activities
from rest.backends.ml_backend.artifact_export import write_artifact_to_bigquery
from rest.config.artifact_export import (
    ArtifactExportInput,
    ArtifactOutTable,
    NamedOutputConfig,
)
from rest.config.batch_prediction_job import OutputArtifactStats
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.jobs import WriteMode
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import ArtifactExportError


@pytest.mark.parametrize('artifact_type', [
    PredictionArtifactType.PREDICTIONS,
    PredictionArtifactType.EMBEDDINGS,
])
@pytest.mark.parametrize('multiple_roots', [False, True])
def test_bigquery_exports_exact_selected_parts_without_root_inference(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: PredictionArtifactType,
    multiple_roots: bool,
) -> None:
    """The BigQuery request must contain all selected parts, never the root.

    Keep selection resolution, activity dispatch, export handling, and the
    connector/wrapper real. Only manifest storage and provider I/O are faked.
    """
    root = f'gs://artifacts/attempt-a/{artifact_type.output_file_name}'
    other_root = f'gs://artifacts/attempt-b/{artifact_type.output_file_name}'
    first = root + '/part-00000-00000.parquet'
    second = other_root + '/part-00001-00000.parquet'
    stale_root = f'gs://artifacts/abandoned/{artifact_type.output_file_name}'
    objects = {
        root: [first],
        other_root: [second],
        stale_root: [stale_root + '/part-00001-00000.parquet'],
    }
    roots = (root, other_root) if multiple_roots else (root, )
    expected_uris = [first, second] if multiple_roots else [first]
    selected = SimpleNamespace(
        artifact_type=artifact_type,
        root=root,
        roots=roots,
        file_count=len(expected_uris),
    )
    monkeypatch.setattr(
        parquet_io_activities, 'load_prediction_selection',
        lambda *_args, **_kwargs: SimpleNamespace(artifacts=(selected, )))
    listed_roots = []

    def list_objects(path):
        listed_roots.append(path)
        return objects[path]

    monkeypatch.setattr(fsutil, 'listdir', list_objects)
    connector = BigQueryConnector('output', project_id='customer-project',
                                  dataset_id='output_dataset')
    provider = Mock(spec=bigquery.Client)
    provider.get_table.side_effect = NotFound('table does not exist')
    provider.load_table_from_uri.return_value.done.return_value = True
    connector._context = provider
    monkeypatch.setattr(parquet_io_activities, 'get_connector_from_spec',
                        lambda _spec: connector)
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

    with (
            patch.object(connector, '_initialize_gcs_client',
                         side_effect=AssertionError('no new GCS client')),
            patch('google.auth.default',
                  side_effect=AssertionError('no ambient credential lookup')),
            patch.object(fsutil, 'copy',
                         side_effect=AssertionError('no local staging')),
    ):
        result = parquet_io_activities._write_artifacts_to_customer(
            input, artifact_stats={artifact_type: stats},
            resolved_connectors={artifact_type: [Mock()]})

    request = provider.load_table_from_uri.call_args.kwargs
    assert request['source_uris'] == expected_uris
    assert request[
        'destination'] == 'customer-project.output_dataset.predictions'
    assert request['job_config'].write_disposition == 'WRITE_APPEND'
    assert request['job_config'].parquet_options.enable_list_inference
    assert listed_roots == list(roots)
    assert result[
        0].output_path == 'customer-project.output_dataset.predictions'
    assert result[0].num_rows == 10
    assert not connector.is_connected


@pytest.mark.parametrize(('path', 'expected_uri'), [
    ('gs://artifacts/predictions.parquet',
     'gs://artifacts/predictions.parquet'),
    ('gs://artifacts/predictions', 'gs://artifacts/predictions/*.parquet'),
    ('gs://artifacts/predictions/', 'gs://artifacts/predictions/*.parquet'),
    ('gs://artifacts/predictions.parquet/',
     'gs://artifacts/predictions.parquet/*.parquet'),
    ('gs://artifacts/predictions/*.parquet',
     'gs://artifacts/predictions/*.parquet'),
])
@pytest.mark.parametrize('replace', [False, True])
def test_bigquery_legacy_uri_handling_and_load_options_are_unchanged(
    path: str,
    expected_uri: str,
    replace: bool,
) -> None:
    connector = BigQueryConnector('output', project_id='customer-project',
                                  dataset_id='output_dataset')
    provider = Mock(spec=bigquery.Client)
    schema = [bigquery.SchemaField('embedding', 'FLOAT', mode='REPEATED')]
    provider.get_table.return_value.schema = schema
    provider.load_table_from_uri.return_value.done.return_value = True
    connector._context = provider

    with patch.object(connector, '_initialize_gcs_client',
                      side_effect=AssertionError('no new GCS client')):
        connector.write_artifacts_to_table('predictions', path, replace=replace)

    request = provider.load_table_from_uri.call_args.kwargs
    assert request['source_uris'] == expected_uri
    config = request['job_config']
    assert config.schema == schema
    assert not config.autodetect
    assert config.parquet_options.enable_list_inference
    assert config.write_disposition == ('WRITE_TRUNCATE'
                                        if replace else 'WRITE_APPEND')
    provider.load_table_from_uri.return_value.result.assert_called_once_with()


@pytest.mark.parametrize('files', [
    [],
    ['s3://artifacts/part.parquet'],
    ['/tmp/part.parquet'],
    ['https://example/part.parquet'],
    ['gs://'],
    ['gs:///part.parquet'],
    ['gs://artifacts'],
    ['gs://artifacts/predictions.parquet/'],
    ['gs://artifacts/predictions.parquet/*.parquet'],
    ['gs://artifacts/part.parquet', 'gs://artifacts/part.parquet'],
])
def test_invalid_selected_files_fail_before_bigquery_io(files) -> None:
    connector = BigQueryConnector('output', project_id='customer-project',
                                  dataset_id='output_dataset')
    provider = Mock(spec=bigquery.Client)
    provider.get_table.side_effect = NotFound('table does not exist')
    provider.load_table_from_uri.return_value.done.return_value = True
    connector._context = provider

    with pytest.raises(ConnectorError, match='selected.*GCS|Selected.*GCS'):
        connector.write_artifacts_to_table(
            'predictions', 'gs://artifacts/predictions.parquet',
            artifact_files=files)

    provider.get_table.assert_not_called()
    provider.load_table_from_uri.assert_not_called()


def test_selected_remote_files_cannot_bypass_local_staging() -> None:
    connector = BigQueryConnector('output', project_id='customer-project',
                                  dataset_id='output_dataset')
    provider = Mock(spec=bigquery.Client)
    provider.get_table.side_effect = NotFound('table does not exist')
    provider.load_table_from_uri.return_value.done.return_value = True
    connector._context = provider

    with pytest.raises(ConnectorError, match='selected.*GCS|Selected.*GCS'):
        connector.write_artifacts_to_table(
            'predictions', '/tmp/predictions.parquet',
            artifact_files=['gs://artifacts/part.parquet'])

    provider.get_table.assert_not_called()
    provider.load_table_from_uri.assert_not_called()


def test_selected_local_bp_output_retains_existing_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'predictions.parquet'
    root.mkdir()
    part = root / 'part-00000-00000.parquet'
    pq.write_table(pa.table({'ID': [1], 'SCORE': [0.5]}), part)
    artifact_type = PredictionArtifactType.PREDICTIONS
    monkeypatch.setattr(
        parquet_io_activities, 'load_prediction_selection',
        lambda *_args, **_kwargs: SimpleNamespace(artifacts=(SimpleNamespace(
            artifact_type=artifact_type, root=str(root), roots=(str(root), ),
            file_count=1), )))
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'staging-bucket')
    connector = BigQueryConnector('output', project_id='customer-project',
                                  dataset_id='output_dataset')
    provider = Mock(spec=bigquery.Client)
    provider.get_table.side_effect = NotFound('table does not exist')
    provider.load_table_from_uri.return_value.done.return_value = True
    connector._context = provider
    gcs_client = Mock()
    monkeypatch.setattr(connector, '_initialize_gcs_client', lambda: gcs_client)
    monkeypatch.setattr(parquet_io_activities, 'get_connector_from_spec',
                        lambda _spec: connector)
    output = NamedOutputConfig(
        connector=ConnectorExecutionRef(
            project_id=ProjectID('project_' + 'a' * 32),
            connector_id=ConnectorID('connector_' + 'b' * 32)),
        out_table=ArtifactOutTable(table_name='predictions',
                                   write_mode=WriteMode.APPEND))
    input = ArtifactExportInput(artifact_paths={artifact_type: str(root)},
                                output_config={artifact_type: [output]},
                                prediction_selection=Mock(),
                                prediction_artifact_types=(artifact_type, ))
    stats = OutputArtifactStats(artifact_type=artifact_type, num_rows=1,
                                num_cols=2, size_bytes=part.stat().st_size,
                                output_path=str(root))

    result = parquet_io_activities._write_artifacts_to_customer(
        input, artifact_stats={artifact_type: stats},
        resolved_connectors={artifact_type: [Mock()]})

    source_uri = provider.load_table_from_uri.call_args.kwargs['source_uris']
    assert source_uri.startswith('gs://staging-bucket/artifact-export/')
    assert source_uri.endswith('/*.parquet')
    blob = gcs_client.bucket.return_value.blob.return_value
    blob.upload_from_filename.assert_called_once_with(
        str(part), content_type='application/octet-stream')
    blob.delete.assert_called_once_with()
    assert result[0].num_rows == 1
    assert not connector.is_connected


@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('load_fails', [False, True])
def test_overwrite_keeps_staging_cleanup_and_sanitizes_load_failures(
    selected: bool,
    load_fails: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = BigQueryConnector('output', project_id='customer-project',
                                  dataset_id='output_dataset')
    provider = Mock(spec=bigquery.Client)
    provider.get_table.return_value.schema = [
        bigquery.SchemaField('embedding', 'FLOAT', mode='REPEATED')
    ]
    load_job = provider.load_table_from_uri.return_value
    load_job.done.return_value = True
    secret = 'provider detail with private-token'
    if load_fails:
        load_job.result.side_effect = Forbidden(secret)
    connector._context = provider
    root = 'gs://artifacts/predictions.parquet'
    files = [root + '/part-00000-00000.parquet'] if selected else None

    def export():
        return write_artifact_to_bigquery(root, connector, WriteMode.OVERWRITE,
                                          'predictions', known_size_bytes=240,
                                          artifact_files=files)

    if load_fails:
        with pytest.raises(ArtifactExportError) as error:
            export()
        assert secret not in str(error.value)
        assert secret not in caplog.text
        provider.query.assert_not_called()
    else:
        result = export()
        assert result.size_bytes == 240
        assert result.display_path == 'customer-project.output_dataset.predictions'
        query = provider.query.call_args.args[0]
        assert 'CREATE OR REPLACE TABLE' in query
        assert 'CLONE' in query
        assert 'predictions_STAGING_' in query

    request = provider.load_table_from_uri.call_args.kwargs
    assert request['source_uris'] == (files if selected else root)
    assert 'predictions_STAGING_' in request['destination']
    assert request['job_config'].write_disposition == 'WRITE_TRUNCATE'
    assert request['job_config'].parquet_options.enable_list_inference
    provider.delete_table.assert_called_once()
