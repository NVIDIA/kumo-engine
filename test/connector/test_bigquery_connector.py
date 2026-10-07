# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import tempfile
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from test.connector.artifact_staging_contract import (
    assert_indexed_parquet_parts,
    local_artifact_dataset,
)
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import google.cloud.bigquery as bigquery
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from google.api_core.exceptions import (
    BadRequest,
    Forbidden,
    GoogleAPIError,
    NotFound,
    PermissionDenied,
)
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype, Stype

import kumo.connector.bigquery_connector as bigquery_connector
import kumo.pquery.AST.ast_node as ast_node
from io_utils import fsutil
from kumo.config.graph import BaseColumnConfig
from kumo.connector.bigquery_connector import (
    ENV_GOOGLE_CREDENTIALS_JSON,
    BigQueryConnector,
    BigQueryOutputSchemaError,
)
from kumo.connector.connector import (
    Connector,
    ConnectorError,
    SourceTableNotFoundError,
)
from kumo.connector.sql_connector import InternalWriteMode
from kumo.stats.compute_stats_python import compute_col_stats_python
from kumo.testing.bigquery import (
    FAIL_PROJECT_PERMISSION_BIGQUERY_KEY,
    FAIL_TABLE_PERMISSION_BIGQUERY_KEY,
    withBigQueryCredentials,
)
from statedb.utils.errors import (
    ConnectorValidationDiagnosticCode,
    sql_query_error_message,
)


def _write_df_to_bigquery(connector, table_name, df, replace=False):
    r"""Test helper: write a pandas DataFrame to a BigQuery table."""
    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        write_disposition=(bigquery.WriteDisposition.WRITE_APPEND if not replace
                           else bigquery.WriteDisposition.WRITE_TRUNCATE),
    )

    job = connector.context.load_table_from_dataframe(
        df, connector.full_table_name(table_name), job_config=job_config)
    job.result()


@withBigQueryCredentials
def test_bigquery_connector_config():
    bq_connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    config = asdict(bq_connector.config())
    assert config == {
        '_target_': 'kumo.connector.bigquery_connector.BigQueryConnector',
        'name': 'bigquery',
        'project_id': 'kumo-engineering',
        'dataset_id': 'financial',
        'location': 'US',
        'billing_project_id': 'kumo-engineering',
    }
    assert bq_connector == Connector.from_config(bq_connector.config())
    assert len(bq_connector.list_tables()) == 8


class FakeGcsClient:
    """An in-memory bucket; ``upload_raises`` maps an object-name suffix to
    the error its upload fails with."""

    def __init__(self, upload_raises=None):
        self.bodies: dict[str, bytes] = {}
        self.uploaded: dict[str, str] = {}
        self.upload_raises = upload_raises or {}

    def bucket(self, name):
        return SimpleNamespace(
            name=name,
            blob=lambda blob_name: _FakeGcsBlob(self, blob_name),
        )

    def list_blobs(self, bucket, prefix=None):
        return [
            SimpleNamespace(name=name)
            for name in sorted(self.bodies)
            if name.startswith(prefix or '')
        ]

    def batch(self):
        return nullcontext()


class _FakeGcsBlob:

    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def upload_from_filename(self, path, content_type=None):
        for suffix, error in self.owner.upload_raises.items():
            if self.name.endswith(suffix):
                raise error
        with open(path, 'rb') as file:
            self.owner.bodies[self.name] = file.read()
        self.owner.uploaded[self.name] = content_type

    def delete(self):
        self.owner.bodies.pop(self.name, None)


def _uploaded_table(gcs: FakeGcsClient, name: str) -> pa.Table:
    return pq.read_table(pa.BufferReader(gcs.bodies[name]))


def _mocked_bigquery_output_connector() -> BigQueryConnector:
    connector = BigQueryConnector(
        name='bigquery',
        project_id='customer-project',
        dataset_id='authorized_dataset',
    )
    connector._context = Mock()
    return connector


def test_bigquery_table_listing_keeps_lightweight_names_sorted():
    connector = _mocked_bigquery_output_connector()
    connector.context.list_tables.return_value = [
        Mock(table_id='orders'),
        Mock(table_id='customers'),
    ]

    assert connector.list_tables() == ['customers', 'orders']
    connector.context.list_tables.assert_called_once_with(
        dataset=connector.dataset_id)
    connector.context.get_table.assert_not_called()


def test_bigquery_missing_table_metadata_is_typed():
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = NotFound('missing')

    with pytest.raises(SourceTableNotFoundError,
                       match="Source table 'missing_table'"):
        connector._get_bigquery_table('missing_table')


def test_bigquery_metadata_ignore_errors_preserves_available_tables():
    connector = _mocked_bigquery_output_connector()
    metadata = Mock(table_name='orders', source_name='orders')
    missing = SourceTableNotFoundError('missing', connector.name)

    with patch.object(
            connector,
            '_get_one_table_metadata',
            side_effect=[missing, metadata],
    ):
        result = connector._get_table_metadata(
            ['missing', 'orders'],
            skip_inferral=True,
            ignore_errors=True,
        )

    assert result == [metadata]


def test_bigquery_metadata_ignore_errors_surfaces_unrelated_failure():
    connector = _mocked_bigquery_output_connector()

    with (
            patch.object(
                connector,
                '_get_one_table_metadata',
                side_effect=ConnectorError('metadata service failed'),
            ),
            pytest.raises(ConnectorError, match='metadata service failed'),
    ):
        connector._get_table_metadata(
            ['orders'],
            skip_inferral=True,
            ignore_errors=True,
        )


def test_bigquery_output_write_probe_is_unique_expiring_and_cleaned_up():
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = NotFound('missing')

    connector.check_table_write_permissions(
        table_name='predictions',
        table_schema='authorized_dataset',
        mode=InternalWriteMode.APPEND,
    )

    probe = connector.context.create_table.call_args.args[0]
    assert probe.table_id.startswith('__kumo_write_probe_')
    assert probe.dataset_id == 'authorized_dataset'
    assert probe.project == 'customer-project'
    assert probe.expires is not None
    connector.context.delete_table.assert_called_once_with(
        probe, not_found_ok=True)
    connector.context.test_iam_permissions.assert_not_called()


def test_bigquery_existing_output_requires_mode_permissions():
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.return_value = Mock(table_type='TABLE')
    connector.context.test_iam_permissions.return_value = {
        'permissions': ['bigquery.tables.get']
    }

    with pytest.raises(ConnectorError, match='requested write mode'):
        connector.check_table_write_permissions(
            table_name='predictions',
            table_schema='authorized_dataset',
            mode=InternalWriteMode.OVERWRITE,
        )

    connector.context.create_table.assert_not_called()
    requested = set(connector.context.test_iam_permissions.call_args.args[1])
    assert requested == {
        'bigquery.tables.get',
        'bigquery.tables.update',
        'bigquery.tables.updateData',
    }


def test_bigquery_existing_output_with_permissions_still_probes_create():
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.return_value = Mock(table_type='TABLE')
    connector.context.test_iam_permissions.return_value = {
        'permissions': [
            'bigquery.tables.get',
            'bigquery.tables.updateData',
        ]
    }

    connector.check_table_write_permissions(
        table_name='predictions',
        table_schema='authorized_dataset',
        mode=InternalWriteMode.APPEND,
    )

    probe = connector.context.create_table.call_args.args[0]
    connector.context.delete_table.assert_called_once_with(
        probe, not_found_ok=True)


def test_bigquery_output_probe_denial_is_sanitized(caplog):
    connector = _mocked_bigquery_output_connector()
    secret = 'provider-payload-with-access-token'
    connector.context.get_table.side_effect = NotFound('missing')
    connector.context.create_table.side_effect = Forbidden(secret)

    with pytest.raises(ConnectorError) as exc_info:
        connector.check_table_write_permissions(
            table_name='predictions',
            table_schema='authorized_dataset',
            mode=InternalWriteMode.APPEND,
        )

    assert secret not in exc_info.value.get_user_message()
    assert secret not in caplog.text
    connector.context.delete_table.assert_not_called()


def test_bigquery_output_probe_cleanup_failure_is_sanitized(caplog):
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = NotFound('missing')
    secret = 'provider-cleanup-payload-with-private-key'
    connector.context.delete_table.side_effect = Forbidden(secret)

    with pytest.raises(ConnectorError, match='cleanup failed') as exc_info:
        connector.check_table_write_permissions(
            table_name='predictions',
            table_schema='authorized_dataset',
            mode=InternalWriteMode.APPEND,
        )

    assert secret not in exc_info.value.get_user_message()
    assert secret not in caplog.text
    connector.context.delete_table.assert_called_once()


@pytest.mark.parametrize(
    ('table_name', 'dataset'),
    [
        ('other.dataset.predictions', 'authorized_dataset'),
        ('predictions', 'other_dataset'),
        ('`other.dataset.predictions`', 'authorized_dataset'),
    ],
)
def test_bigquery_output_rejects_target_outside_connector(
    table_name,
    dataset,
):
    connector = _mocked_bigquery_output_connector()

    with pytest.raises(ConnectorError):
        connector.check_table_write_permissions(
            table_name=table_name,
            table_schema=dataset,
            mode=InternalWriteMode.APPEND,
        )

    connector.context.create_table.assert_not_called()


def test_bigquery_actual_output_write_rejects_qualified_target_before_io():
    connector = _mocked_bigquery_output_connector()

    with pytest.raises(ConnectorError, match='unqualified table ID'):
        connector.write_artifacts_to_table(
            'other_dataset.predictions',
            'gs://internal-artifacts/predictions/',
        )

    connector.context.load_table_from_uri.assert_not_called()


@withBigQueryCredentials
def test_bigquery_read_table():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    df = connector.read_table('LOAN', pkey=BaseColumnConfig('LOAN_ID'))
    assert df.shape == (682, 6)

    chunk = connector.read_table('LOAN', pkey=BaseColumnConfig('LOAN_ID'),
                                 num_rows=5)
    assert df[:5].equals(chunk)

    chunk = connector.read_table('LOAN', pkey=BaseColumnConfig('LOAN_ID'),
                                 num_rows=3, offset=2)
    assert df[2:5].equals(chunk)


def test_bigquery_s3_artifact_output_fails_before_credentials_or_io(caplog):
    connector = _mocked_bigquery_output_connector()
    secret_path = 's3://secret-bucket/private/access-token/part-0.parquet'

    with (
            patch('google.auth.default', side_effect=AssertionError) as adc,
            patch.object(
                connector,
                '_use_credentials',
                side_effect=AssertionError,
            ) as secret_lookup,
            patch.object(
                connector,
                '_initialize_gcs_client',
                side_effect=AssertionError,
            ) as gcs_client,
            pytest.raises(
                ConnectorError,
                match='does not support s3://',
            ) as exc_info,
    ):
        connector.write_artifacts_to_table('predictions', secret_path)

    adc.assert_not_called()
    secret_lookup.assert_not_called()
    gcs_client.assert_not_called()
    connector.context.load_table_from_uri.assert_not_called()
    assert secret_path not in exc_info.value.get_user_message()
    assert secret_path not in caplog.text


def test_bigquery_gcs_artifact_output_keeps_direct_load_path():
    connector = _mocked_bigquery_output_connector()

    with patch.object(connector, '_write_artifacts_from_gcs') as gcs_load:
        connector.write_artifacts_to_table(
            'predictions', 'gs://internal-artifacts/predictions/', replace=True)

    gcs_load.assert_called_once_with('predictions',
                                     'gs://internal-artifacts/predictions/',
                                     replace=True)


@pytest.mark.parametrize(
    'embeddings',
    [False, True],
    ids=['predictions', 'embeddings'],
)
def test_bigquery_local_artifact_stages_with_brokered_gcs_client(
        monkeypatch, tmp_path, embeddings):
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'internal-transfer')
    artifact_path = local_artifact_dataset(tmp_path, shards=2,
                                           embeddings=embeddings)
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = NotFound('missing')
    gcs = FakeGcsClient()
    loaded = {}

    def load(source_uris, destination, job_config):
        directory = source_uris.removeprefix('gs://internal-transfer/')
        directory = directory.removesuffix('part-*.parquet')
        names = [blob.name for blob in gcs.list_blobs('b', prefix=directory)]
        loaded.update(
            uris=source_uris, config=job_config, destination=destination,
            names=names,
            rows=sum(_uploaded_table(gcs, name).num_rows for name in names))
        return Mock()

    connector.context.load_table_from_uri.side_effect = load
    with patch.object(connector, '_initialize_gcs_client', return_value=gcs):
        connector.write_artifacts_to_table('predictions', str(artifact_path),
                                           replace=True)

    prefix = assert_indexed_parquet_parts(list(gcs.uploaded))
    assert prefix.startswith('artifact-export/')
    assert set(gcs.uploaded.values()) == {'application/octet-stream'}
    assert loaded['uris'] == f'gs://internal-transfer/{prefix}/part-*.parquet'
    assert loaded['names'] == sorted(gcs.uploaded)
    assert loaded['rows'] == 2
    assert loaded['destination'] == (
        'customer-project.authorized_dataset.predictions')
    config = loaded['config']
    assert config.write_disposition == 'WRITE_TRUNCATE'
    assert config.parquet_options.enable_list_inference
    assert not config.autodetect
    assert [(f.name, f.mode) for f in config.schema] == [
        ('ENTITY_ID', 'NULLABLE'),
        ('EMBEDDING' if embeddings else 'SCORE',
         'REPEATED' if embeddings else 'NULLABLE'),
    ]
    assert gcs.bodies == {}


def test_bigquery_local_artifact_append_conforms_to_existing_table(
        monkeypatch, tmp_path):
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'internal-transfer')
    artifact_path = local_artifact_dataset(tmp_path)
    connector = _mocked_bigquery_output_connector()
    existing = Mock(table_type='TABLE', schema=[
        bigquery.SchemaField('ENTITY_ID', 'INT64'),
        bigquery.SchemaField('SCORE', 'FLOAT64'),
    ])
    connector.context.get_table.return_value = existing

    with patch.object(connector, '_initialize_gcs_client',
                      return_value=FakeGcsClient()):
        connector.write_artifacts_to_table('predictions', str(artifact_path))

    config = connector.context.load_table_from_uri.call_args.kwargs[
        'job_config']
    assert config.schema == existing.schema
    assert config.write_disposition == 'WRITE_APPEND'


def test_bigquery_local_artifact_append_rejects_incompatible_table(
        monkeypatch, tmp_path, caplog):
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'internal-transfer')
    artifact_path = local_artifact_dataset(tmp_path)
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.return_value = Mock(
        table_type='TABLE',
        schema=[bigquery.SchemaField('ENTITY_ID', 'STRING')])
    gcs = FakeGcsClient()

    with (
            patch.object(connector, '_initialize_gcs_client',
                         return_value=gcs),
            pytest.raises(BigQueryOutputSchemaError) as error,
    ):
        connector.write_artifacts_to_table('predictions', str(artifact_path))

    detail = ('ENTITY_ID is INTEGER NULLABLE but the table column is STRING '
              'NULLABLE; SCORE (FLOAT NULLABLE) is not a table column')
    assert detail in error.value.message
    assert detail in caplog.text

    connector.context.load_table_from_uri.assert_not_called()
    assert gcs.uploaded == {}
    assert gcs.bodies == {}


@pytest.mark.parametrize('scenario', ['partial_upload', 'load_failure'])
def test_bigquery_local_artifact_failure_cleanup_contract(
        monkeypatch, tmp_path, caplog, scenario):
    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'internal-transfer')
    artifact_path = local_artifact_dataset(tmp_path, shards=2)
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = NotFound('missing')
    gcs = FakeGcsClient()
    if scenario == 'partial_upload':
        gcs.upload_raises['part-00000001.parquet'] = RuntimeError(
            'provider detail')
    else:
        connector.context.load_table_from_uri.return_value.result.\
            side_effect = BadRequest('provider detail')

    with (
            patch.object(connector, '_initialize_gcs_client',
                         return_value=gcs),
            pytest.raises(ConnectorError) as exc_info,
    ):
        connector.write_artifacts_to_table('predictions', str(artifact_path))

    assert 'Failed to load local artifacts' in str(exc_info.value)
    assert 'provider detail' not in caplog.text
    assert 'provider detail' not in str(exc_info.value)
    assert gcs.bodies == {}
    if scenario == 'partial_upload':
        assert list(gcs.uploaded) == [
            n for n in gcs.uploaded if n.endswith('part-00000000.parquet')
        ]
        connector.context.load_table_from_uri.assert_not_called()
    else:
        assert len(gcs.uploaded) == 2


def test_bigquery_local_artifact_requires_configured_transfer_bucket(
        monkeypatch, tmp_path):
    monkeypatch.delenv('GCP_STORAGE_TRANSFER_BUCKET', raising=False)
    artifact_path = local_artifact_dataset(tmp_path)
    connector = _mocked_bigquery_output_connector()

    with (
            patch.object(connector, '_initialize_gcs_client') as gcs_client,
            pytest.raises(
                ConnectorError,
                match='GCP_STORAGE_TRANSFER_BUCKET is not configured'),
    ):
        connector.write_artifacts_to_table('predictions', str(artifact_path))

    gcs_client.assert_not_called()


def test_bigquery_gcs_artifact_output_waits_for_load_job():
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = NotFound('missing')
    load_job = connector.context.load_table_from_uri.return_value

    connector._write_artifacts_from_gcs('predictions',
                                        'gs://internal-artifacts/predictions/')

    load_job.result.assert_called_once_with()


def test_bigquery_gcs_artifact_load_failure_is_sanitized(caplog):
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = NotFound('missing')
    secret = 'provider-schema-error-with-private-key'
    load_job = connector.context.load_table_from_uri.return_value
    load_job.result.side_effect = GoogleAPIError(secret)

    with pytest.raises(ConnectorError, match='Failed to load artifacts') as exc:
        connector._write_artifacts_from_gcs(
            'predictions', 'gs://internal-artifacts/predictions/')

    assert exc.value.__cause__ is None
    assert secret not in exc.value.get_user_message()
    assert secret not in caplog.text


@withBigQueryCredentials
def test_bigquery_overwrite_table():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    assert connector.table_exists("customers")
    assert connector.table_exists("articles")
    overwrite_table_name = f"test_{uuid.uuid4().hex}"

    try:
        # Test overwrite to a new table.
        connector.overwrite_table("customers", overwrite_table_name)
        assert connector._get_table_num_rows(
            "customers") == connector._get_table_num_rows(overwrite_table_name)

        # Test overwrite existing table.
        connector.overwrite_table("articles", overwrite_table_name)
        assert connector._get_table_num_rows(
            "articles") == connector._get_table_num_rows(overwrite_table_name)
    finally:
        connector.drop_table_if_exists(overwrite_table_name)
        assert not connector.table_exists(overwrite_table_name)


@withBigQueryCredentials
def test_get_exact_col_stats():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )

    out = connector._get_exact_col_stats(
        table_name='LOAN',
        stats_dict={
            BaseColumnConfig.cast('AMOUNT'): [
                ColStatType.MEAN,
                ColStatType.MEDIAN,
                ColStatType.MIN,
                ColStatType.MAX,
                ColStatType.STD,
            ],
            BaseColumnConfig.cast('STATUS'): [
                ColStatType.NUM_UNIQUE,
                ColStatType.UNIQUE_FRACTION,
                ColStatType.CATEGORY_COUNTS,
            ],
        },
    )
    assert len(out) == 2

    stats = out[BaseColumnConfig.cast('AMOUNT')]
    assert len(stats) == 5
    assert stats[ColStatType.MEAN].value == pytest.approx(151410.1)
    assert stats[ColStatType.MEAN].num_entries == 682
    assert stats[ColStatType.MEDIAN].value == 116928
    assert stats[ColStatType.MEDIAN].num_entries == 682
    assert stats[ColStatType.MIN].value == 4980
    assert stats[ColStatType.MIN].num_entries == 682
    assert stats[ColStatType.MAX].value == 590820
    assert stats[ColStatType.MAX].num_entries == 682
    assert stats[ColStatType.STD].value == pytest.approx(113372.4)
    assert stats[ColStatType.STD].num_entries == 682

    stats = out[BaseColumnConfig.cast('STATUS')]
    assert len(stats) == 3
    assert stats[ColStatType.NUM_UNIQUE].value == 4
    assert stats[ColStatType.NUM_UNIQUE].num_entries == 682
    assert stats[ColStatType.UNIQUE_FRACTION].value == pytest.approx(
        0.0058, abs=1e-2)
    assert stats[ColStatType.UNIQUE_FRACTION].num_entries == 682
    assert stats[ColStatType.CATEGORY_COUNTS].value == [('C', 403), ('A', 203),
                                                        ('D', 45), ('B', 31)]
    assert stats[ColStatType.CATEGORY_COUNTS].num_entries == 682

    out = connector._get_exact_col_stats(
        'DISTRICT',
        {
            BaseColumnConfig.cast('A12'): [
                ColStatType.NUM_NA,
                ColStatType.NA_FRACTION,
            ],
        },
    )
    assert len(out) == 1

    stats = out[BaseColumnConfig.cast('A12')]
    assert stats[ColStatType.NUM_NA].value == 0
    assert stats[ColStatType.NUM_NA].num_entries == 77
    # TODO: The district table in BigQuery financial dataset is marked as '?'
    # instead of a NULL value. Hence NA_FRACTION=0.
    # assert stats[ColStatType.NA_FRACTION].value ==
    # pytest.approx(1 / 77, 0.001)
    # assert stats[ColStatType.NA_FRACTION].num_entries == 77
    assert stats[ColStatType.NA_FRACTION].value == 0
    assert stats[ColStatType.NA_FRACTION].num_entries == 77


@withBigQueryCredentials
def test_bigquery_data_version():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    expected_table_data_versions = {
        'TRANS': 1695079444,
        'ORDERS': 1695077884,
    }
    for table_name, version in expected_table_data_versions.items():
        assert connector._get_data_version(table_name) == str(version)


@withBigQueryCredentials
def test_bigquery_table_info():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )

    # num rows
    assert connector._get_table_num_rows('ACCOUNT') == 4500
    assert connector._get_table_num_rows('CARD') == 892
    assert connector._get_table_num_rows('CLIENT') == 5369
    assert connector._get_table_num_rows('DISP') == 5369
    assert connector._get_table_num_rows('DISTRICT') == 77
    assert connector._get_table_num_rows('LOAN') == 682
    assert connector._get_table_num_rows('ORDERS') == 6471

    # table size (in bytes)
    assert connector._get_table_source_size('ACCOUNT') == 188706
    assert connector._get_table_source_size('CARD') == 37055
    assert connector._get_table_source_size('CLIENT') == 144963
    assert connector._get_table_source_size('DISP') == 169915
    assert connector._get_table_source_size('DISTRICT') == 10317
    assert connector._get_table_source_size('LOAN') == 34782
    assert connector._get_table_source_size('ORDERS') == 270796


@withBigQueryCredentials
def test_bigquery_table_dtype():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )

    metadata = connector._get_one_table_metadata('embeddings2')
    assert metadata.cols[0].name == 'article_id'
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].name == 'embedding'
    assert metadata.cols[1].dtype == Dtype.floatlist

    metadata = connector._get_one_table_metadata('embeddings')
    assert metadata.cols[0].name == 'article_id'
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].name == 'embedding'
    # Return unsupported data type because it is a record.
    assert metadata.cols[1].dtype == Dtype.unsupported


@withBigQueryCredentials
def test_bigquery_connector_list_types(monkeypatch):
    monkeypatch.setattr(
        ast_node,
        'ALLOW_ARRAY_TARGETS',
        True,
    )

    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='movielens',
    )
    metadata = connector.get_table_metadata(["movie_stringlist"])[0]
    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.stringlist

    metadata = connector.get_table_metadata(["movie_intlist"])[0]
    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.intlist


# BigQuery doesn't have NA sequence columns but just empty lists.
# Hence, we test the stats of a sequence column with a few empty lists.
@withBigQueryCredentials
def test_emb_col_stats():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='test',
    )
    key_col = BaseColumnConfig('A', stype=Stype.numerical, dtype=Dtype.int64)
    emb_col = BaseColumnConfig('EMB', stype=Stype.sequence, dtype=Dtype.intlist)
    connector.open()
    df = connector.read_table('TEST_EMPTY_EMB', cols=[key_col, emb_col])
    stats = compute_col_stats_python(df['EMB'], [
        ColStatType.SEQUENCE_MAX_LENGTH,
        ColStatType.SEQUENCE_MIN_LENGTH,
        ColStatType.NUM_NA,
    ])
    assert stats[ColStatType.SEQUENCE_MAX_LENGTH].value == 3
    assert stats[ColStatType.SEQUENCE_MIN_LENGTH].value == 3
    assert stats[ColStatType.NUM_NA].value == 0


@withBigQueryCredentials
def test_bq_non_existent_table_data_version():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # The table is legit - no error should be thrown.
    connector._get_data_version('LOAN')

    with pytest.raises(SourceTableNotFoundError) as missing_table_error:
        connector._get_data_version('NON_EXISTENT_TABLE')
    message = str(missing_table_error.value)
    assert "Source table 'NON_EXISTENT_TABLE'" in message


@withBigQueryCredentials
def test_read_table_sample():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # Since BigQuery sampling operation is not deterministic, we check for
    # approximate correctness
    # Sample with num_rows:
    sample_df = connector.read_table_sample(table_name='LOAN', num_rows=100)
    assert sample_df.shape[0] == pytest.approx(100, rel=0.1)
    # Sample with frac:
    sample_df = connector.read_table_sample(table_name='LOAN', frac=0.1)
    total_num_rows = connector._get_table_num_rows('LOAN') * 1.0
    assert sample_df.shape[0] / total_num_rows == pytest.approx(0.1, abs=1e-2)


@withBigQueryCredentials
def test_get_table_num_rows():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    total_rows = connector._get_table_num_rows("transactions")
    assert total_rows == 31788324

    view_total_rows = connector._get_table_num_rows("transactions_view")
    assert view_total_rows == total_rows


@withBigQueryCredentials
def test_set_credentials_and_open():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # This will get credentials from the JSON key file.
    connector.open()
    # Corrupt the env variable so that we don't get credentials from there:
    os.environ[ENV_GOOGLE_CREDENTIALS_JSON] = "random"

    connector2 = BigQueryConnector(
        name='bigquery2',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # Test the workflow of explicitly setting credentials and opening the
    # connector.
    connector2.set_credentials(connector._credentials)
    connector.open()


@withBigQueryCredentials(service_account_key=FAIL_TABLE_PERMISSION_BIGQUERY_KEY)
def test_bigquery_table_permission_fail_read_table():
    connector = BigQueryConnector(
        name='bigquery_fail',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    with pytest.raises(ConnectorError):
        connector.check_table_read_permissions(table_name='LOAN')
    with pytest.raises(ConnectorError) as caught:
        connector.read_table(table_name='LOAN', num_rows=100)
    message = str(caught.value)
    assert message == (
        'The selected identity does not have the required BigQuery dataset '
        'read permissions.')
    assert 'LOAN' not in message


@withBigQueryCredentials(
    service_account_key=FAIL_PROJECT_PERMISSION_BIGQUERY_KEY)
def test_bigquery_project_permission_fail_read_table():
    connector = BigQueryConnector(
        name='bigquery_fail',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    with pytest.raises(ConnectorError):
        connector.check_table_read_permissions(table_name='LOAN')
    with pytest.raises(ConnectorError, match=r"does not have .+ permission"):
        connector.read_table(table_name='LOAN', num_rows=100)


def test_bigquery_empty_credentials_and_open():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='financial',
    )
    # Empty explicit credentials use the deployment's stored-credential path.
    # Keep this unit test independent of AWS Secrets Manager availability.
    with patch.object(
            connector, '_use_credentials',
            side_effect=ConnectorError('stored credentials unavailable')):
        with pytest.raises(
                ConnectorError,
                match='An error occured while trying to connect to BigQuery'):
            connector.open()


def test_bigquery_opens_with_activity_local_google_credential_object():
    credential = Mock(name='google-credential')
    client = Mock(name='bigquery-client')
    connector = BigQueryConnector(
        name='bigquery',
        project_id='customer-data-project',
        dataset_id='events',
        location='EU',
        billing_project_id='billing-project',
    )
    connector.set_credentials({
        'provider': 'bigquery',
        'credential': credential,
    })

    with patch('kumo.connector.bigquery_connector.Client',
               return_value=client) as constructor:
        assert connector._open_with_credentials() is client

    constructor.assert_called_once_with(
        project='customer-data-project',
        credentials=credential,
        location='EU',
    )


def test_bigquery_rejects_cross_provider_runtime_credential_safely():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='customer-data-project',
        dataset_id='events',
    )
    connector.set_credentials({
        'provider': 'gcs',
        'credential': Mock(token='poison-token'),
    })

    with pytest.raises(ConnectorError, match='another provider') as caught:
        connector._open_with_credentials()

    assert 'poison-token' not in str(caught.value)


def test_bigquery_read_failure_hides_provider_text_target_and_logs(caplog):
    poison = 'poison-token private-read-path'
    connector = _mocked_bigquery_output_connector()
    connector.context.get_table.side_effect = GoogleAPIError(poison)

    with pytest.raises(ConnectorError) as caught:
        connector._get_bigquery_table('private-read-path')

    public_message = str(caught.value)
    assert 'Kumo could not read BigQuery table metadata' in public_message
    assert poison not in public_message
    assert 'private-read-path' not in public_message
    assert poison not in caplog.text
    assert 'private-read-path' not in caplog.text
    assert caught.value.__suppress_context__ is True


def test_bigquery_write_failure_hides_provider_text_target_and_logs(caplog):
    poison = 'poison-token gs://private-bucket/private-write-path'
    connector = _mocked_bigquery_output_connector()
    connector.context.query.side_effect = RuntimeError(poison)

    with pytest.raises(ConnectorError) as caught:
        connector.export_table_to_blob_storage(
            'private-write-path',
            'gs://private-bucket/private-write-path',
        )

    public_message = str(caught.value)
    assert 'Failed to export data from the configured BigQuery table' in (
        public_message)
    assert poison not in public_message
    assert 'private-bucket' not in public_message
    assert 'private-write-path' not in public_message
    assert poison not in caplog.text
    assert 'private-bucket' not in caplog.text
    assert 'private-write-path' not in caplog.text
    assert caught.value.__suppress_context__ is True


class TestValidatePermissions:
    def test_validate_permissions_no_tables_found(self):
        """Test validation fails when dataset has no tables."""
        # Arrange
        mock_instance = Mock()
        mock_instance.project_id = "test-project"
        mock_instance.dataset_id = "empty_dataset"

        mock_real_client = Mock()
        mock_instance.client._client = mock_real_client

        # Mock empty result from INFORMATION_SCHEMA query
        mock_query_job = Mock()
        mock_query_job.result.return_value = []
        mock_real_client.query.return_value = mock_query_job

    def test_validate_permissions_forbidden_access_denied(self, caplog):
        """A 403 whose structured reason is ``accessDenied`` is a genuine
        permission failure and classifies as TARGET_ACCESS_DENIED."""
        # Arrange
        mock_instance = Mock()
        mock_instance.project_id = "test-project"
        mock_instance.dataset_id = "restricted_dataset"

        mock_real_client = Mock()
        mock_instance.client._client = mock_real_client

        # Mock Forbidden exception carrying the structured access reason.
        provider_secret = 'signed-provider-request-secret'
        mock_real_client.query.side_effect = Forbidden(
            f'Access denied: {provider_secret}', errors=[{
                'reason':
                'accessDenied'
            }])

        # Act & Assert
        with pytest.raises(ConnectorError) as exc_info:
            BigQueryConnector.validate_permissions(mock_instance)

        message = str(exc_info.value)
        assert message == (
            'The selected identity does not have the required BigQuery '
            'dataset read permissions.')
        assert exc_info.value.diagnostic_code == (
            ConnectorValidationDiagnosticCode.TARGET_ACCESS_DENIED)
        assert 'INFORMATION_SCHEMA.TABLES' not in message
        assert 'restricted_dataset' not in message
        assert 'test-project' not in message
        assert provider_secret not in message
        assert provider_secret not in caplog.text

    @pytest.mark.parametrize(
        'reason',
        [
            'rateLimitExceeded',
            'quotaExceeded',
            'billingNotEnabled',
            'userRateLimitExceeded',
            'responseTooLarge',
            None,  # a 403 with no structured reason must also stay unclassified
        ])
    def test_validate_permissions_forbidden_operational_is_unclassified(
            self, reason, caplog):
        """Operational 403s (quota/rate-limit/billing) and reason-less 403s must
        not be mislabeled as a non-retriable target-access problem."""
        # Arrange
        mock_instance = Mock()
        mock_instance.project_id = "test-project"
        mock_instance.dataset_id = "restricted_dataset"

        mock_real_client = Mock()
        mock_instance.client._client = mock_real_client

        provider_secret = 'signed-provider-request-secret'
        errors = [{'reason': reason}] if reason is not None else []
        mock_real_client.query.side_effect = Forbidden(
            f'Quota exceeded: {provider_secret}', errors=errors)

        # Act & Assert
        with pytest.raises(ConnectorError) as exc_info:
            BigQueryConnector.validate_permissions(mock_instance)

        message = str(exc_info.value)
        assert message == 'BigQuery connector access validation failed.'
        assert exc_info.value.diagnostic_code is None
        assert provider_secret not in message
        assert provider_secret not in caplog.text

    def test_validate_permissions_grpc_permission_denied(self, caplog):
        """A gRPC/IAM PERMISSION_DENIED is unambiguously access-denied."""
        # Arrange
        mock_instance = Mock()
        mock_instance.project_id = "test-project"
        mock_instance.dataset_id = "restricted_dataset"

        mock_real_client = Mock()
        mock_instance.client._client = mock_real_client

        provider_secret = 'signed-provider-request-secret'
        mock_real_client.query.side_effect = PermissionDenied(
            f'Permission denied: {provider_secret}')

        # Act & Assert
        with pytest.raises(ConnectorError) as exc_info:
            BigQueryConnector.validate_permissions(mock_instance)

        message = str(exc_info.value)
        assert message == (
            'The selected identity does not have the required BigQuery '
            'dataset read permissions.')
        assert exc_info.value.diagnostic_code == (
            ConnectorValidationDiagnosticCode.TARGET_ACCESS_DENIED)
        assert provider_secret not in message
        assert provider_secret not in caplog.text

    def test_validate_permissions_dataset_not_found(self):
        """Test validation fails when dataset doesn't exist."""
        # Arrange
        mock_instance = Mock()
        mock_instance.project_id = "test-project"
        mock_instance.dataset_id = "nonexistent_dataset"

        mock_real_client = Mock()
        mock_instance.client._client = mock_real_client

        # Mock NotFound exception
        mock_real_client.query.side_effect = NotFound("Dataset not found")

        # Act & Assert
        with pytest.raises(ConnectorError) as exc_info:
            BigQueryConnector.validate_permissions(mock_instance)

        message = str(exc_info.value)
        assert message == (
            'The configured BigQuery dataset was not found or is not '
            'accessible to the selected identity.')
        assert exc_info.value.diagnostic_code == (
            ConnectorValidationDiagnosticCode.TARGET_NOT_FOUND)
        assert 'nonexistent_dataset' not in message
        assert 'test-project' not in message


def test_bigquery_validate_permissions_timeout_fails_closed():
    timeout_const = (
        bigquery_connector.BIGQUERY_CONNECTOR_PERMISSIONS_CHECK_TIMEOUT)
    connector = BigQueryConnector(
        name='bigquery',
        project_id='data-project',
        dataset_id='events',
    )
    connector._context = Mock()
    fake_future = Mock()
    fake_future.result.side_effect = FuturesTimeoutError('timed out')
    fake_executor = Mock()
    fake_executor.__enter__ = Mock(return_value=fake_executor)
    fake_executor.__exit__ = Mock(return_value=False)
    fake_executor.submit = Mock(return_value=fake_future)

    with patch('kumo.connector.bigquery_connector.ThreadPoolExecutor',
               return_value=fake_executor):
        with pytest.raises(ConnectorError, match='validation timed out'):
            connector.validate_permissions()

    fake_executor.submit.assert_called_once()
    fake_future.result.assert_called_once_with(timeout=timeout_const)
    fake_future.cancel.assert_called_once_with()
    fake_executor.shutdown.assert_called_once_with(wait=False,
                                                   cancel_futures=True)


@withBigQueryCredentials
def test_invalid_project():
    with pytest.raises(ConnectorError) as caught:
        connector = BigQueryConnector(
            name='bigquery',
            project_id='MXYZPTLK',
            dataset_id='hnm_integ_test',
            location='us-west1',
        )
        connector.open()
    message = str(caught.value)
    assert message == 'BigQuery connector access validation failed.'
    assert 'MXYZPTLK' not in message
    assert 'hnm_integ_test' not in message


@withBigQueryCredentials
def test_invalid_dataset():
    with pytest.raises(ConnectorError) as caught:
        connector = BigQueryConnector(
            name='bigquery',
            project_id='kumo-engineering',
            dataset_id='mxyzptlk',
        )
        connector.open()
    message = str(caught.value)
    assert message == (
        'The configured BigQuery dataset was not found or is not accessible '
        'to the selected identity.')
    assert 'mxyzptlk' not in message
    assert 'kumo-engineering' not in message


# ==========================================================================
# export_table_to_blob_storage integration tests
# ==========================================================================
_BQ_EXPORT_TEST_STAGING_PATH = os.environ.get(
    'BIGQUERY_EXPORT_TEST_STAGING_PATH',
    'gs://kumo-test-bucket/bigquery-integ-test/')


@withBigQueryCredentials
def test_export_table_to_blob_storage():
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_export_{uuid.uuid4().hex}"
    dest_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                         f'export_{uuid.uuid4().hex}/')

    nrows = 100
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
        'STATUS': ['A'] * nrows,
    })
    _write_df_to_bigquery(connector, table_name, source_df, replace=True)

    try:
        connector.export_table_to_blob_storage(table_name, dest_path)

        exported_files = fsutil.listdir(dest_path, suffix='parquet')
        assert exported_files, f"No parquet files at {dest_path}"

        exported = pd.concat([pd.read_parquet(f) for f in exported_files])
        exported = exported.sort_values('LOAN_ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(exported[source_df.columns], source_df,
                                      check_dtype=False)
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(dest_path, ignore_errors=True)


@withBigQueryCredentials
def test_export_table_to_blob_storage_with_columns():
    r"""Functional test: exporting with the ``columns`` argument should
    produce parquet with exactly the requested columns (and none of the
    others). Asserts that unselected columns are absent from the output
    schema.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_export_cols_{uuid.uuid4().hex}"
    dest_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                         f'export_cols_{uuid.uuid4().hex}/')

    nrows = 50
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
        'STATUS': ['A'] * nrows,
        'EXTRA_COL': ['should_not_appear'] * nrows,
    })
    _write_df_to_bigquery(connector, table_name, source_df, replace=True)

    try:
        connector.export_table_to_blob_storage(table_name, dest_path,
                                               columns=['LOAN_ID', 'AMOUNT'])

        exported_files = fsutil.listdir(dest_path, suffix='parquet')
        assert exported_files, f"No parquet files at {dest_path}"

        exported = pd.concat([pd.read_parquet(f) for f in exported_files])
        exported = exported.sort_values('LOAN_ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(
            exported[['LOAN_ID', 'AMOUNT']],
            source_df[['LOAN_ID', 'AMOUNT']],
            check_dtype=False,
        )
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(dest_path, ignore_errors=True)


@withBigQueryCredentials
def test_export_table_to_blob_storage_error_nonexistent_table():
    r"""Exporting a table that doesn't exist must raise ConnectorError
    (not silently succeed).
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    missing_table = f"does_not_exist_{uuid.uuid4().hex}"
    dest_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                         f'missing_{uuid.uuid4().hex}/')

    try:
        with pytest.raises(ConnectorError):
            connector.export_table_to_blob_storage(missing_table, dest_path)
    finally:
        # Defensive cleanup in case the export partially succeeded.
        fsutil.rm(dest_path, ignore_errors=True)


@pytest.mark.parametrize('columns', [None, ['ID', 'column with spaces']])
def test_export_table_to_blob_storage_uses_shared_direct_export(columns):
    from kumo.connector.bigquery_connector import WrappedClient

    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.return_value.total_bytes_processed = 123
    with patch.object(connector, '_source_reader',
                      side_effect=AssertionError('local materialization')), \
            patch.object(WrappedClient, '_get_query_source_size_internal'), \
            patch.object(WrappedClient, 'poll_job_completion') as poll:
        assert connector.export_table_to_blob_storage('orders',
                                                      'gs://bucket/final/',
                                                      columns=columns) is None
    query = connector.context.query.call_args.kwargs['query']
    poll.assert_called_once_with(connector.context.query.return_value,
                                 WrappedClient.BQ_INGEST_TIMEOUT_SECONDS, query)
    assert "uri='gs://bucket/final/part-*.parquet'" in query
    projection = '*' if columns is None else '`ID`, `column with spaces`'
    assert f'AS SELECT {projection} FROM `test-project`.`test_dataset`.`orders`' in query
    connector.context.query.return_value.result.assert_called_once_with()


def test_export_table_to_blob_storage_rejects_non_gcs_before_query():
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    with pytest.raises(ConnectorError, match='expected gs://'):
        connector.export_table_to_blob_storage('orders', 's3://bucket/final')
    connector.context.query.assert_not_called()


# ==========================================================================
# write_artifacts_to_table gs:// fast-path integration tests
# ==========================================================================
# These tests exercise the single-hop GCS -> BigQuery load path
# (``load_table_from_uri``). Each test writes parquet directly to a
# Kumo-owned GCS bucket, calls ``write_artifacts_to_table`` with a gs://
# path, then reads the BigQuery table back and asserts the data landed
# correctly. Cross-cloud s3:// output is intentionally rejected before
# authentication or I/O by the focused tests above.


def _seed_gcs_parquet(df: pd.DataFrame, gcs_path: str) -> None:
    r"""Write ``df`` to a single parquet file under ``gcs_path`` so the
    gs:// fast path has something to load.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_parquet(osp.join(tmpdir, 'part-0.parquet'), index=False)
        fsutil.copy(tmpdir, gcs_path)


@withBigQueryCredentials
def test_bigquery_write_artifacts_to_table_from_gcs():
    r"""Functional test: seed parquet in a Kumo-owned GCS bucket, then
    load it into a fresh BigQuery table via the gs:// fast path. Verifies
    the rows and columns round-trip.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_gcs_write_{uuid.uuid4().hex}"
    gcs_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                        f'gcs_write_{uuid.uuid4().hex}/')

    nrows = 100
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
        'STATUS': ['A'] * nrows,
    })
    try:
        _seed_gcs_parquet(source_df, gcs_path)

        connector.write_artifacts_to_table(table_name, gcs_path)

        loaded = connector.read_table(table_name)
        loaded = loaded.sort_values('LOAN_ID').reset_index(drop=True)
        pd.testing.assert_frame_equal(loaded[source_df.columns], source_df,
                                      check_dtype=False)
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(gcs_path, ignore_errors=True)


@withBigQueryCredentials
def test_bigquery_write_artifacts_to_table_from_gcs_with_embedding_column():
    r"""Verify pandas-generated parquet with an embedding (list) column
    round-trips through BigQuery's ``enable_list_inference=True``.

    This test uses ``pandas.to_parquet``, matching the artifact writer used
    by the current pipeline. Succeeding here means BigQuery's list inference
    recognizes its standard parquet LIST logical type.

    Failure here means the comment "only works for PySpark generated
    parquet files" is empirically load-bearing and the gs:// fast path
    can't be used with pandas-written embedding columns until we add
    a conversion step.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_gcs_embed_{uuid.uuid4().hex}"
    gcs_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                        f'gcs_embed_{uuid.uuid4().hex}/')

    nrows = 50
    emb_size = 5
    source_df = pd.DataFrame({
        'LOAN_ID':
        list(range(nrows)),
        'EMBEDDING':
        [[float(i + k) for k in range(emb_size)] for i in range(nrows)],
    })
    try:
        _seed_gcs_parquet(source_df, gcs_path)
        connector.write_artifacts_to_table(table_name, gcs_path)

        # ARRAY_LENGTH only succeeds when EMBEDDING is a BQ ARRAY
        # column. Without enable_list_inference (or if it's ignored),
        # BQ would read the parquet LIST as a nested STRUCT and this
        # query would fail with a SQL type error.
        query = (f'SELECT COUNT(*) AS n FROM '
                 f'{connector.full_table_name(table_name)} '
                 f'WHERE ARRAY_LENGTH(EMBEDDING) = {emb_size}')
        result_df = connector.context.query(query).result().to_dataframe()
        assert result_df['n'].iloc[0] == nrows, (
            f"Expected {nrows} rows with EMBEDDING of length {emb_size}; "
            f"got {result_df['n'].iloc[0]}. pandas-generated LIST parquet "
            f"may not be read as a BQ ARRAY via enable_list_inference.")
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(gcs_path, ignore_errors=True)


@withBigQueryCredentials
def test_bigquery_write_artifacts_to_table_from_gcs_append_and_replace():
    r"""Functional test: the ``replace`` parameter controls
    ``WriteDisposition``. Seeds parquet, writes once (append on a new
    table), writes again with ``replace=False`` and confirms rows double,
    then writes again with ``replace=True`` and confirms rows are
    truncated back to the original count.
    """
    connector = BigQueryConnector(
        name='bigquery',
        project_id='kumo-engineering',
        dataset_id='hnm_integ_test',
        location='us-west1',
    )
    table_name = f"test_gcs_replace_{uuid.uuid4().hex}"
    gcs_path = osp.join(_BQ_EXPORT_TEST_STAGING_PATH,
                        f'gcs_replace_{uuid.uuid4().hex}/')

    nrows = 50
    source_df = pd.DataFrame({
        'LOAN_ID': list(range(nrows)),
        'AMOUNT': [float(i) * 100 for i in range(nrows)],
    })
    try:
        _seed_gcs_parquet(source_df, gcs_path)

        # 1st write: creates the table, loads nrows rows.
        connector.write_artifacts_to_table(table_name, gcs_path)
        assert connector._get_table_num_rows(table_name) == nrows

        # 2nd write (replace=False): appends, row count doubles.
        connector.write_artifacts_to_table(table_name, gcs_path, replace=False)
        assert connector._get_table_num_rows(table_name) == 2 * nrows

        # 3rd write (replace=True): truncates, back to nrows.
        connector.write_artifacts_to_table(table_name, gcs_path, replace=True)
        assert connector._get_table_num_rows(table_name) == nrows
    finally:
        connector.drop_table_if_exists(table_name)
        fsutil.rm(gcs_path, ignore_errors=True)


# =============================================================================
# Diskgraph ingest path: export_query / download unit tests (no credentials, no
# network I/O — patched at module boundaries). These exercise the failure
# branches that the existing integration tests don't reach.
# =============================================================================


def _make_unopened_bq_connector() -> BigQueryConnector:
    """Construct a BigQueryConnector without opening a real client.

    Construction doesn't perform I/O; only `client`/`context` lazily
    authenticates. The sad-path tests here patch any code that would
    otherwise touch the client.
    """
    return BigQueryConnector(
        name='bigquery',
        project_id='test-project',
        dataset_id='test_dataset',
    )


def _bq_read_client(schema: pa.Schema, pages: dict) -> Mock:
    """A Storage Read client serving `pages` (stream name -> record batches)."""
    session = Mock()
    session.arrow_schema.serialized_schema = schema.serialize().to_pybytes()
    streams = []
    for stream_name in pages:
        stream = Mock()
        stream.name = stream_name
        streams.append(stream)
    session.streams = streams

    read_client = Mock()
    read_client.create_read_session.return_value = session

    def _read_rows(stream_name):
        messages = []
        for batch in pages[stream_name]:
            sink = pa.BufferOutputStream()
            options = pa.ipc.IpcWriteOptions(compression='zstd')
            with pa.ipc.new_stream(sink, schema, options=options) as writer:
                writer.write_batch(batch)
            encoded = list(pa.ipc.MessageReader.open_stream(sink.getvalue()))
            message = Mock()
            message.arrow_record_batch.serialized_record_batch = (
                encoded[1].serialize().to_pybytes())
            messages.append(message)
        return iter(messages)

    read_client.read_rows.side_effect = _read_rows
    return read_client


@contextmanager
def _query_export(connector, schema, pages, destination):
    read_client = _bq_read_client(schema, pages)
    job = Mock(
        total_bytes_processed=123, statement_type='SELECT',
        schema=[bigquery.SchemaField(name, 'INTEGER') for name in schema.names])
    job.ddl_target_table = bigquery.TableReference.from_string(
        'result-project.session.query_result')
    job.session_info.session_id = 'session-id'
    connector._context = Mock()
    connector._context.query.return_value = job

    def materialize_path(fingerprint, version, stage):
        return str(destination /
                   f'{fingerprint.ingest_path_fpid}-{version}-{stage.name}')

    with patch.dict(os.environ, {'GCP_STORAGE_TRANSFER_BUCKET': ''}), \
            patch('kumo.connector.bigquery_connector.BigQueryReadClient',
                  return_value=read_client), \
            patch('kumo_bigquery.streaming.read_clients',
                  side_effect=lambda count, **kwargs: (read_client,) * count), \
            patch('kumo.connector.source_bridge.get_materialize_path',
                  side_effect=materialize_path), \
            patch('kumo.data.utils.get_materialize_path',
                  side_effect=materialize_path):
        yield read_client


@pytest.mark.parametrize('billing_project', [None, 'billing-project'])
@pytest.mark.parametrize('empty', [False, True])
def test_bigquery_export_query_then_download_reads_that_artifact(
        tmp_path, billing_project, empty):
    connector = BigQueryConnector(name='bigquery', project_id='test-project',
                                  dataset_id='test_dataset',
                                  billing_project_id=billing_project)
    schema = pa.schema([('ID', pa.int64())])
    pages = {} if empty else {
        'stream-0': [pa.record_batch({'ID': [1, 2]})],
        'stream-1': [pa.record_batch({'ID': [3]})],
    }
    table_fp = Mock(ingest_path_fpid='fpid_sql', cols={'ID': Dtype.int})
    destination, downloaded = tmp_path / 'persisted', tmp_path / 'downloaded'
    sql = 'SELECT id AS ID FROM orders WHERE amount > 0; -- user comment'

    with _query_export(connector, schema, pages, destination) as read_client:
        metrics = connector.export_query(sql, table_fp, '9')
        connector.download('orders', table_fp, '9', str(downloaded))

    calls = connector.context.query.call_args_list
    assert calls[0].args[0] == sql
    assert calls[0].kwargs['job_config'].dry_run
    assert calls[-2].args[0] == (
        'CREATE TEMP TABLE _SESSION.kumo_read_target AS '
        'SELECT `ID` FROM (SELECT id AS ID FROM orders WHERE amount > 0)')
    assert calls[-2].kwargs['job_config'].create_session
    assert calls[-1].args[0] == 'DROP TABLE _SESSION.kumo_read_target'
    properties = calls[-1].kwargs['job_config'].connection_properties
    assert [(p.key, p.value)
            for p in properties] == [('session_id', 'session-id')]
    request = read_client.create_read_session.call_args.kwargs
    assert request['parent'] == f'projects/{billing_project or "test-project"}'
    assert request['read_session'].table == (
        'projects/result-project/datasets/session/tables/query_result')
    assert request['read_session'].read_options.selected_fields == ['ID']
    parts = list(downloaded.glob('*.parquet'))
    result = pa.concat_tables([pq.read_table(part) for part in parts])
    assert result.schema == schema
    assert sorted(result['ID'].to_pylist()) == ([] if empty else [1, 2, 3])
    assert metrics.num_rows == (0 if empty else 3)
    assert metrics.source_size == sum(p.stat().st_size for p in parts)


@pytest.mark.parametrize('phase',
                         ['create_read_session', 'read_rows', 'publish'])
def test_bigquery_export_query_cleans_partial_on_failure(tmp_path, phase):
    connector = _make_unopened_bq_connector()
    schema = pa.schema([('ID', pa.int64())])
    pages = {'stream-0': [pa.record_batch({'ID': [1]})]}
    table_fp = Mock(ingest_path_fpid='fpid_sql', cols={'ID': Dtype.int})
    destination = tmp_path / 'persisted'
    failure = RuntimeError('private provider detail')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    with _query_export(connector, schema, pages, destination) as read_client, \
            patch('kumo.connector.source_bridge._scratch_dir',
                  return_value=str(scratch)), \
            patch('kumo.connector.source_bridge.publish_source',
                  side_effect=failure) if phase == 'publish' else nullcontext():
        if phase != 'publish':
            getattr(read_client, phase).side_effect = failure
        with pytest.raises(ConnectorError) as error:
            connector.export_query('SELECT id AS ID FROM t', table_fp, '9')

    assert connector.context.query.call_args.args[0] == (
        'DROP TABLE _SESSION.kumo_read_target')
    read_client.transport.close.assert_called_once_with()
    assert not scratch.exists()
    assert not destination.exists()
    assert 'private provider detail' not in error.value.get_user_message()


class _FakeGCSClient:
    """Objects keyed by name, listed and deleted by prefix like GCS."""
    def __init__(self, names):
        self.names = set(names)

    def list_blobs(self, bucket, prefix=''):
        for name in sorted(self.names):
            if name.startswith(prefix):
                yield self._blob(name)

    def bucket(self, name):
        return SimpleNamespace(blob=self._blob)

    def batch(self):
        return nullcontext()

    def _blob(self, name):
        return SimpleNamespace(name=name,
                               delete=lambda: self.names.discard(name))


def test_bigquery_source_reader_stages_each_attempt_separately(monkeypatch):
    """``staged_fetch`` clears its location before the unload and again in
    ``finally``, so on a shared prefix one attempt's cleanup would delete the
    other's export out from under its fetch.
    """
    from kumo_bigquery.staging import StagingLocation, clear_prefix

    monkeypatch.setenv('GCP_STORAGE_TRANSFER_BUCKET', 'example-bucket')
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    table_fp = Mock(ingest_path_fpid='fpid_abc')

    with patch.object(connector, '_initialize_gcs_client',
                      return_value=Mock()), \
            patch('kumo_bigquery.BigQuerySourceReader') as reader, \
            patch('kumo.connector.bigquery_connector.BigQueryReadClient'):
        with connector._source_reader(table_fp.ingest_path_fpid, '3'):
            pass
        with connector._source_reader(table_fp.ingest_path_fpid, '3'):
            pass

    first, second = [
        call.kwargs['staging_uri'] for call in reader.call_args_list
    ]
    root = 'gs://example-bucket/source-reader/fpid_abc/3/'
    assert first.startswith(root) and second.startswith(root)
    assert first != second

    # Each attempt exports one object under its own prefix; the clear that
    # brackets either read leaves the other attempt's object untouched.
    locations = [
        StagingLocation.parse(uri).child('orders') for uri in (first, second)
    ]
    objects = [f'{location.directory}part-0.parquet' for location in locations]
    storage = _FakeGCSClient(objects)
    clear_prefix(storage, locations[0])
    assert storage.names == {objects[1]}


def test_bigquery_export_propagates_timeout_as_connector_error():
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.side_effect = TimeoutError('private timeout')
    with pytest.raises(ConnectorError) as excinfo:
        connector.export_table_to_blob_storage('LOAN', 'gs://bucket/export/')
    assert 'Failed to export data' in str(excinfo.value)
    assert 'private timeout' not in str(excinfo.value)
    assert excinfo.value.__suppress_context__ is True


def test_bigquery_export_query_needs_no_gcs_staging_authority(tmp_path):
    connector = _make_unopened_bq_connector()
    schema = pa.schema([('ID', pa.int64())])
    pages = {'stream-0': [pa.record_batch({'ID': [1]})]}
    table_fp = Mock(ingest_path_fpid='fpid_sql', cols={'ID': Dtype.int})
    with _query_export(connector, schema, pages, tmp_path / 'persisted'), \
            patch('google.auth.default', side_effect=AssertionError('ADC')), \
            patch('google.cloud.storage.Client',
                  side_effect=AssertionError('GCS staging')), \
            patch.object(connector, '_use_credentials',
                         side_effect=AssertionError('internal credentials')):
        assert connector.export_query('SELECT 1 AS ID', table_fp,
                                      '9').num_rows == 1


@pytest.mark.parametrize(
    'credentials',
    [
        {},
        {
            'provider': 'bigquery'
        },
        {
            'provider': 'gcs',
            'credential': Mock(name='wrong_provider_credential'),
        },
    ],
)
def test_bigquery_staging_rejects_missing_or_wrong_provider_credentials(
        credentials):
    """Staging fails closed instead of falling back to ambient authority."""
    connector = _make_unopened_bq_connector()
    connector.set_credentials(credentials)

    with patch('google.cloud.storage.Client') as storage_client:
        with pytest.raises(
                ConnectorError,
                match='Authorized BigQuery staging credentials are unavailable',
        ) as excinfo:
            connector._initialize_gcs_client()

    storage_client.assert_not_called()
    assert excinfo.value.__cause__ is None


# Custom SQL query-compile error classification ###############################
#
# A BigQuery dry-run/compile failure is a ``BadRequest`` (HTTP 400) whose
# payload is a diagnostic about the user's OWN SQL — safe and actionable to
# surface. Auth/permission failures (``Forbidden``/403) stay redacted. These
# tests pin that discriminator across all three Custom SQL query primitives and
# confirm the composed user-facing message the authoring UI shows.

_REDACTION_TAIL = 'Verify the selected identity'
_REDACTION_BODY = 'provider rejected'


def _bad_query_error() -> BadRequest:
    return BadRequest(
        'Unrecognized name: a at [1:8]',
        errors=[{
            'message': 'Unrecognized name: a at [1:8]',
            'reason': 'invalidQuery',
        }],
    )


# A non-``invalidQuery`` HTTP 400 whose text carries account/request detail that
# must NOT be surfaced (e.g. a backend/internal or invalid-user error).
_NON_QUERY_400_DETAIL = 'internal-account-detail-42'


def _non_query_bad_request() -> BadRequest:
    return BadRequest(
        _NON_QUERY_400_DETAIL,
        errors=[{
            'message': _NON_QUERY_400_DETAIL,
            'reason': 'jobInternalError',
        }],
    )


def test_bigquery_get_query_metadata_bad_request_surfaces_reason():
    """A dry-run compile ``BadRequest`` surfaces the raw compile diagnostic
    (no ``error=``), so the redaction never fires and the UI sees the real
    reason."""
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.side_effect = _bad_query_error()

    with pytest.raises(ConnectorError) as exc_info:
        connector.get_query_metadata('SELECT a')

    msg = exc_info.value.get_user_message()
    assert 'Unrecognized name' in msg
    assert _REDACTION_TAIL not in msg
    assert _REDACTION_BODY not in msg
    # Full composition the authoring UI renders (CustomSqlQueryError body).
    assert sql_query_error_message(exc_info.value) == (
        'The warehouse could not run this query. '
        'BigQuery could not run the query: Unrecognized name: a at [1:8]')


def test_bigquery_get_query_metadata_forbidden_stays_redacted():
    """A ``Forbidden`` still routes through ``error=`` → the Google-error
    redaction, so provider text never leaks and the permission hint remains."""
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    secret = 'provider-payload-with-access-token'
    connector.context.query.side_effect = Forbidden(secret)

    with pytest.raises(ConnectorError) as exc_info:
        connector.get_query_metadata('SELECT 1')

    msg = exc_info.value.get_user_message()
    assert secret not in msg
    assert _REDACTION_TAIL in msg
    composed = sql_query_error_message(exc_info.value)
    assert secret not in composed
    assert _REDACTION_TAIL in composed


def test_bigquery_get_query_metadata_not_found_has_location_guidance():
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.side_effect = NotFound('secret provider detail')
    with pytest.raises(ConnectorError) as caught:
        connector.get_query_metadata('SELECT * FROM missing')
    assert 'configured location' in caught.value.get_user_message()
    assert 'secret provider detail' not in caught.value.get_user_message()


@pytest.mark.parametrize('statement_type', ['SCRIPT', 'CALL', 'INSERT'])
def test_bigquery_rejects_non_select_statement_classification(statement_type):
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.return_value = Mock(statement_type=statement_type,
                                                schema=[])

    with pytest.raises(ConnectorError, match='single read-only SELECT'):
        connector.get_query_metadata('warehouse-specific syntax')


def test_bigquery_get_query_scan_estimate_bad_request_surfaces_reason():
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.side_effect = _bad_query_error()

    with pytest.raises(ConnectorError) as exc_info:
        connector.get_query_scan_estimate_bytes('SELECT a')

    msg = exc_info.value.get_user_message()
    assert 'Unrecognized name' in msg
    assert _REDACTION_TAIL not in msg
    assert _REDACTION_BODY not in msg


def test_bigquery_get_query_scan_estimate_forbidden_stays_redacted():
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    secret = 'provider-payload-with-access-token'
    connector.context.query.side_effect = Forbidden(secret)

    with pytest.raises(ConnectorError) as exc_info:
        connector.get_query_scan_estimate_bytes('SELECT 1')

    msg = exc_info.value.get_user_message()
    assert secret not in msg
    assert _REDACTION_TAIL in msg


def test_bigquery_sample_query_bad_request_surfaces_reason():
    """``sample_query`` reads via ``_get_df_result`` (shared with physical
    reads); ``reraise_bad_request`` lets the compile ``BadRequest`` propagate
    so the real reason is surfaced, not the generic read-failure message."""
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    mock_client = Mock()
    mock_client.query.side_effect = _bad_query_error()

    with patch.object(BigQueryConnector, 'client', new_callable=PropertyMock,
                      return_value=mock_client):
        with pytest.raises(ConnectorError) as exc_info:
            connector.sample_query('SELECT a', 10)

    msg = exc_info.value.get_user_message()
    assert 'Unrecognized name' in msg
    assert _REDACTION_TAIL not in msg
    assert _REDACTION_BODY not in msg
    assert 'failed while reading table data' not in msg
    assert sql_query_error_message(exc_info.value) == (
        'The warehouse could not run this query. '
        'BigQuery could not run the query: Unrecognized name: a at [1:8]')


def test_bigquery_sample_query_forbidden_stays_redacted():
    """A ``Forbidden`` during the preview keeps ``_get_df_result``'s sanitized
    permission message — provider text never leaks and the reason is not a raw
    compile diagnostic."""
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    secret = 'provider-payload-with-access-token'
    mock_client = Mock()
    mock_client.query.side_effect = Forbidden(secret)

    with patch.object(BigQueryConnector, 'client', new_callable=PropertyMock,
                      return_value=mock_client):
        with pytest.raises(ConnectorError) as exc_info:
            connector.sample_query('SELECT 1', 10)

    msg = exc_info.value.get_user_message()
    assert secret not in msg
    assert 'required BigQuery' in msg


def test_bigquery_sample_query_carries_a_ceiling_the_warehouse_enforces():
    from kumo.data.custom_sql import SQL_SAMPLE_MAX_SCAN_BYTES

    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    mock_client = Mock()
    mock_client.query.side_effect = RuntimeError('stop after the job config')

    with patch.object(BigQueryConnector, 'client', new_callable=PropertyMock,
                      return_value=mock_client):
        with pytest.raises(ConnectorError):
            connector.sample_query('SELECT 1', 10)

    job_config = mock_client.query.call_args.kwargs['job_config']
    assert job_config.maximum_bytes_billed == SQL_SAMPLE_MAX_SCAN_BYTES


def test_bigquery_is_the_warehouse_whose_samples_are_costed_first():
    from kumo.connector.connector import Connector

    assert BigQueryConnector.gates_query_scan is True
    assert Connector.gates_query_scan is False


def test_bigquery_sample_query_generic_error_stays_masked():
    """A non-BadRequest execution error keeps the generic read-failure message
    (provider text stays masked)."""
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    secret = 'internal-transport-detail'
    mock_client = Mock()
    mock_client.query.side_effect = RuntimeError(secret)

    with patch.object(BigQueryConnector, 'client', new_callable=PropertyMock,
                      return_value=mock_client):
        with pytest.raises(ConnectorError) as exc_info:
            connector.sample_query('SELECT 1', 10)

    msg = exc_info.value.get_user_message()
    assert secret not in msg
    assert 'BigQuery failed while reading table data' in msg


@pytest.mark.parametrize('operation', ['metadata', 'estimate', 'preview'])
def test_bigquery_structured_resource_error_is_actionable_and_redacted(
        operation):
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    secret = 'private recursive planner detail'
    error = BadRequest(
        secret, errors=[{
            'reason': 'resourcesExceeded',
            'message': secret,
        }])
    if operation == 'preview':
        client = Mock()
        client.query.side_effect = error
        context = patch.object(BigQueryConnector, 'client',
                               new_callable=PropertyMock, return_value=client)
    else:
        connector.context.query.side_effect = error
        context = nullcontext()
    with context, pytest.raises(ConnectorError) as caught:
        if operation == 'metadata':
            connector.get_query_metadata('WITH RECURSIVE ...')
        elif operation == 'estimate':
            connector.get_query_scan_estimate_bytes('WITH RECURSIVE ...')
        else:
            connector.sample_query('WITH RECURSIVE ...', 10)
    message = caught.value.get_user_message()
    assert 'resource limits' in message
    assert 'Simplify the query' in message
    assert secret not in message


def test_bigquery_safe_query_diagnostic_allowlists_not_found_only():
    secret = 'private provider detail'
    diagnostic = bigquery_connector._bq_safe_query_diagnostic(NotFound(secret))
    assert 'configured location' in diagnostic
    assert secret not in diagnostic
    assert bigquery_connector._bq_safe_query_diagnostic(
        BadRequest(secret, errors=[{
            'reason': 'jobInternalError',
            'message': secret,
        }])) is None


def test_bigquery_wrapped_preview_dry_run_preserves_safe_diagnostic():
    secret = 'private recursive planner detail'
    raw_client = Mock()
    raw_client.query.side_effect = BadRequest(
        secret, errors=[{
            'reason': 'resourcesExceeded',
            'message': secret,
        }])
    with pytest.raises(ConnectorError) as caught:
        bigquery_connector.WrappedClient(raw_client).query(
            'WITH RECURSIVE ...', surface_query_diagnostics=True)
    assert 'resource limits' in caught.value.get_user_message()
    assert secret not in caught.value.get_user_message()


def test_bq_query_error_detail_prefers_structured_then_falls_back():
    """The reason extractor prefers the structured compile diagnostic, then the
    exception ``message``, then ``str`` — defensively, even when ``errors`` is
    missing or empty."""
    assert bigquery_connector._bq_query_error_detail(
        _bad_query_error()) == 'Unrecognized name: a at [1:8]'
    # No structured errors → fall back to .message (stripped).
    assert bigquery_connector._bq_query_error_detail(
        BadRequest('  Syntax error near X  ')) == 'Syntax error near X'
    # Empty errors list → still fall back to .message.
    assert bigquery_connector._bq_query_error_detail(
        BadRequest('Table not found', errors=[])) == 'Table not found'


def test_bigquery_get_query_metadata_non_query_400_stays_redacted():
    """A non-``invalidQuery`` HTTP 400 routes through ``error=`` → the
    Google-error redaction, so its account/request text never leaks."""
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.side_effect = _non_query_bad_request()

    with pytest.raises(ConnectorError) as exc_info:
        connector.get_query_metadata('SELECT 1')

    msg = exc_info.value.get_user_message()
    assert _NON_QUERY_400_DETAIL not in msg
    assert _REDACTION_TAIL in msg
    assert _NON_QUERY_400_DETAIL not in sql_query_error_message(exc_info.value)


def test_bigquery_sample_query_non_query_400_stays_redacted():
    """A non-``invalidQuery`` HTTP 400 on the preview path re-raised from
    ``_get_df_result`` is redacted, not surfaced raw."""
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    mock_client = Mock()
    mock_client.query.side_effect = _non_query_bad_request()

    with patch.object(BigQueryConnector, 'client', new_callable=PropertyMock,
                      return_value=mock_client):
        with pytest.raises(ConnectorError) as exc_info:
            connector.sample_query('SELECT 1', 10)

    msg = exc_info.value.get_user_message()
    assert _NON_QUERY_400_DETAIL not in msg
    assert _REDACTION_TAIL in msg


# Exercise Google client decoding of REST metadata with omitted scale as
# well as Kumo's mapping; this fixture does not query the provider itself.
@pytest.mark.parametrize(
    'bq_type,precision,scale,dtype',
    [
        # A stated precision with no scale is BigQuery saying "scale zero",
        # and a width int64 certainly holds.
        ('NUMERIC', '18', None, Dtype.int64),
        ('BIGNUMERIC', '18', None, Dtype.int64),
        ('DECIMAL', '18', None, Dtype.int64),
        ('BIGDECIMAL', '18', None, Dtype.int64),
        # Scale zero, but wider than int64 can certainly hold. Declaring
        # int64 here would have DiskGraph drop every out-of-range row off an
        # inference the caller never made.
        ('NUMERIC', '19', None, Dtype.float64),
        ('NUMERIC', '29', None, Dtype.float64),
        ('BIGNUMERIC', '38', None, Dtype.float64),
        ('BIGDECIMAL', '38', None, Dtype.float64),
        # A stated scale above zero is fractional.
        ('NUMERIC', '38', '9', Dtype.float64),
        ('BIGNUMERIC', '50', '20', Dtype.float64),
        ('DECIMAL', '10', '2', Dtype.float64),
        # Unparameterized: the type's default scale applies (9 for NUMERIC,
        # 38 for BIGNUMERIC), so the column is fractional.
        ('NUMERIC', None, None, Dtype.float64),
        ('BIGNUMERIC', None, None, Dtype.float64),
        ('DECIMAL', None, None, Dtype.float64),
        ('BIGDECIMAL', None, None, Dtype.float64),
    ])
def test_bigquery_decimal_types_are_scale_aware(bq_type, precision, scale,
                                                dtype):
    api_repr = {'name': 'amount', 'type': bq_type, 'mode': 'NULLABLE'}
    if precision is not None:
        api_repr['precision'] = precision
    if scale is not None:
        api_repr['scale'] = scale
    field = bigquery.SchemaField.from_api_repr(api_repr)

    assert field.precision == (int(precision)
                               if precision is not None else None)
    assert field.scale == (int(scale) if scale is not None else None)
    assert BigQueryConnector.to_kumo_dtype(field.field_type, field.mode,
                                           field.precision,
                                           field.scale) is dtype


def test_bigquery_decimal_columns_carry_their_parameters_into_inference():
    r"""``_cols_from_schema`` has to forward precision and scale; reading the
    type name alone reads every decimal as a float."""
    connector = _mocked_bigquery_output_connector()
    schema = [
        bigquery.SchemaField.from_api_repr({
            'name': 'merchant_key',
            'type': 'BIGNUMERIC',
            'mode': 'NULLABLE',
            'precision': '18',
        }),
        bigquery.SchemaField.from_api_repr({
            'name': 'loan_amount',
            'type': 'NUMERIC',
            'mode': 'NULLABLE',
            'precision': '38',
            'scale': '9',
        }),
    ]

    cols = connector._cols_from_schema(schema)

    assert [col.dtype for col in cols] == [Dtype.int64, Dtype.float64]


@pytest.mark.parametrize('bq_type', ['BIGNUMERIC', 'DECIMAL', 'BIGDECIMAL'])
def test_bigquery_repeated_decimals_stay_unsupported(bq_type):
    r"""BigQuery exports a repeated decimal uncast, and DiskGraph's
    ``float_list_view`` reads only Float16/32/64 child arrays -- a
    ``list<decimal128|256>`` fails there with "float array type mismatch".
    Calling these float arrays would trade a clear "unsupported" at table
    create for an ingest failure."""
    assert BigQueryConnector.to_kumo_dtype(bq_type, 'REPEATED', 38,
                                           None) is Dtype.unsupported


def test_bigquery_repeated_numeric_keeps_its_existing_mapping():
    r"""``NUMERIC`` has always mapped this way; it is left alone so this
    change adds no new behaviour to the repeated path."""
    assert BigQueryConnector.to_kumo_dtype('NUMERIC', 'REPEATED', None,
                                           None) is Dtype.floatlist


#: Every type name either mapping names, plus spellings neither does. A
#: `kumo_bigquery` column reads its dtype from `dtype_for`, a hosted read
#: reads it from `to_kumo_dtype`, and a graph built by one and ingested by the
#: other must agree about every one of them.
_BIGQUERY_TYPES = [
    'STRING',
    'INTEGER',
    'INT64',
    'FLOAT',
    'FLOAT64',
    'BOOLEAN',
    'BOOL',
    'DATE',
    'TIMESTAMP',
    'DATETIME',
    'TIME',
    'BYTES',
    'NUMERIC',
    'BIGNUMERIC',
    'DECIMAL',
    'BIGDECIMAL',
    'GEOGRAPHY',
    'JSON',
    'RECORD',
    'STRUCT',
    'INTERVAL',
]

#: The decimal metadata that decides int64 against float64: unparameterized,
#: scale zero inside int64's range, scale zero beyond it, and fractional.
_PRECISION_AND_SCALE = [(None, None), (18, None), (29, None), (38, None),
                        (18, 2), (10, 0)]


@pytest.mark.parametrize('bq_type', _BIGQUERY_TYPES)
@pytest.mark.parametrize('mode', ['NULLABLE', 'REQUIRED', 'REPEATED'])
@pytest.mark.parametrize('precision,scale', _PRECISION_AND_SCALE)
def test_the_sdk_plugin_maps_bigquery_types_exactly_as_the_connector_does(
        bq_type, mode, precision, scale):
    from kumo_bigquery.types import dtype_for

    assert dtype_for(bq_type, mode, precision, scale) is \
        BigQueryConnector.to_kumo_dtype(bq_type, mode, precision, scale)


@pytest.mark.parametrize('field_type,mode,expected', [
    ('INT64', 'NULLABLE', Dtype.int64),
    ('int64', 'NULLABLE', Dtype.int64),
    (' INT64 ', 'NULLABLE', Dtype.int64),
    ('INT64', 'repeated', Dtype.intlist),
    ('INT64', ' REPEATED ', Dtype.intlist),
])
def test_bigquery_hosted_type_name_normalization(field_type, mode, expected):
    from kumo_bigquery.types import dtype_for

    assert BigQueryConnector.to_kumo_dtype(field_type, mode) is expected
    assert dtype_for(field_type, mode) is expected


def test_bigquery_scan_estimate_does_not_require_select_classification():
    connector = _make_unopened_bq_connector()
    connector._context = Mock()
    connector.context.query.return_value = Mock(statement_type='SCRIPT',
                                                total_bytes_processed=123)
    assert connector.get_query_scan_estimate_bytes('SELECT 1; SELECT 2') == 123
