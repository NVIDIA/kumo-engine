# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kumoapi.data_source import (
    S3ConnectorResourceConfig,
    S3RootedConnectorTarget,
)
from kumoapi.jobs import (
    ADLSPredictionTableSource,
    GCSPredictionTableSource,
    S3PredictionTableSource,
)
from kumoapi.source_table import (
    ADLSURISource,
    GCSRelativeSource,
    GCSURISource,
    S3RelativeSource,
    S3URISource,
)

from kumo.connector.any_uri_adls_connector import AnyURIADLSConnector
from kumo.connector.any_uri_s3_connector import AnyURIS3Connector
from kumo.connector.composite_file_connector import CompositeFileConnector
from rest.backends.ml_backend.activities import custom_prediction_table
from rest.config.connector import (
    object_store_source_for_connector,
    object_store_source_for_runtime_connector,
)
from rest.config.data_backend import ConnectorExecutionRef
from statedb.config.common import ConnectorID, ProjectID
from statedb.utils.errors import InvalidPredictionTableError
from temporalio_commons import dataclass_converter

PROJECT_ID = ProjectID('project_' + 'a' * 32)
CONNECTOR_ID = ConnectorID('connector_' + 'b' * 32)
REFERENCE = ConnectorExecutionRef(project_id=PROJECT_ID,
                                  connector_id=CONNECTOR_ID)
SOURCE_CASES = [
    (
        S3PredictionTableSource(
            connector_id=str(CONNECTOR_ID), source=S3URISource(
                uri='s3://customer-data/predictions/table.parquet')),
        AnyURIS3Connector,
        's3',
        False,
        None,
    ),
    (
        ADLSPredictionTableSource(
            connector_id=str(CONNECTOR_ID), source=ADLSURISource(
                uri=('abfss://data@account.dfs.core.windows.net/'
                     'predictions/table.parquet'))),
        AnyURIADLSConnector,
        'adls',
        True,
        None,
    ),
    (
        GCSPredictionTableSource(
            connector_id=str(CONNECTOR_ID),
            source=GCSRelativeSource(path='predictions/table.parquet')),
        CompositeFileConnector,
        'gcs',
        False,
        'gs://customer-data/root',
    ),
]
SOURCE = SOURCE_CASES[0][0]


def _connector(connector_type, provider: str, source_path: str, file_type: str,
               scope_state: dict[str, bool], root_dir: str | None = None):
    connector = MagicMock(spec=connector_type)
    connector._provider_name = provider
    if root_dir is not None:
        connector.root_dir = root_dir
    connector.get_path_and_type.return_value = (source_path, file_type)

    @contextmanager
    def scope(*_args):
        scope_state['active'] = True
        try:
            yield
        finally:
            scope_state['active'] = False

    connector.filesystem_credential_scope.side_effect = scope
    return connector


@pytest.mark.parametrize(
    ('source', 'connector_type', 'provider', 'directory_dataset', 'root_dir'),
    SOURCE_CASES,
    ids=['s3-any-uri-file', 'adls-any-uri-directory', 'gcs-rooted-file'],
)
def test_custom_prediction_source_stages_under_connector_scope(
        tmp_path, source, connector_type, provider, directory_dataset,
        root_dir):
    scope_state = {'active': False}
    source_path = (source.source.uri
                   if hasattr(source.source, 'uri') else source.source.path)
    connector = _connector(connector_type, provider, source_path, 'parquet',
                           scope_state, root_dir)
    destination = str(tmp_path / 'durable' / 'custom-prediction')

    @contextmanager
    def local_source(name, *, transform):
        from kumo_core.sources.types import ReadMode, StagedTable
        assert name == source_path
        assert transform is False
        local_path = tmp_path / 'prepared'
        local_path.mkdir()
        part = local_path / 'part-0.parquet'
        pd.DataFrame({'id': [1], 'TIMESTAMP': [0]}).to_parquet(part)
        yield StagedTable(directory=str(local_path), paths=(str(part), ),
                          version='test', source_size=part.stat().st_size,
                          mode_used=ReadMode.STREAMED), 'parquet'

    connector._local_source.side_effect = local_source
    with patch(
            'rest.backends.ml_backend.activities.custom_prediction_table.'
            'get_connector_from_spec', return_value=connector):
        result = custom_prediction_table.stage_custom_prediction_table(
            REFERENCE, source, destination)

    assert result == destination
    assert pd.read_parquet(destination).to_dict('list') == {
        'id': [1],
        'TIMESTAMP': [0],
    }
    connector.close.assert_called_once_with()
    connector.get_path_and_type.assert_called_once_with(source_path)
    assert not scope_state['active']


def test_custom_prediction_source_rejects_csv_before_copy(tmp_path):
    connector = _connector(
        AnyURIS3Connector,
        's3',
        SOURCE.source.uri,
        'csv',
        {'active': False},
        None,
    )
    with (
            patch(
                'rest.backends.ml_backend.activities.custom_prediction_table.'
                'get_connector_from_spec', return_value=connector),
            pytest.raises(InvalidPredictionTableError,
                          match='configured S3 source'),
    ):
        custom_prediction_table.stage_custom_prediction_table(
            REFERENCE, SOURCE, str(tmp_path / 'destination'))

    connector._local_source.assert_not_called()
    connector.close.assert_called_once_with()


def test_custom_prediction_workflow_coordinate_is_credential_free():
    payload = dataclass_converter.DataclassJsonPayloadConverter().to_payload(
        [REFERENCE, SOURCE])
    assert b'project_' in payload.data
    assert b'connector_' in payload.data
    assert b's3://customer-data/predictions/table.parquet' in payload.data
    assert b'credentials' not in payload.data
    assert b'access_key' not in payload.data
    assert b'secret' not in payload.data


def test_any_uri_activity_rejects_source_provider_mismatch_before_io():
    connector = AnyURIADLSConnector(name='adls')
    source = GCSPredictionTableSource(
        connector_id=str(CONNECTOR_ID),
        source=GCSURISource(uri='gs://customer-data/table.parquet'),
    )

    with pytest.raises(ValueError, match='does not match'):
        object_store_source_for_runtime_connector(connector, source.source)


def test_s3_source_uses_shared_root_normalization():
    config = S3ConnectorResourceConfig(
        target=S3RootedConnectorTarget(bucket='customer-data', prefix='root'))

    assert object_store_source_for_connector(
        config,
        's3://customer-data/root/predictions/table.parquet',
    ) == S3RelativeSource(path='predictions/table.parquet')


def test_empty_warehouse_export_needs_no_parquet_files(tmp_path):
    from kumo.connector.connector import ExportMetrics
    from kumo.connector.snowflake_connector import SnowflakeConnector
    from statedb.utils.errors import EmptyPredictionTableError

    connector = MagicMock(spec=SnowflakeConnector)
    connector.export_table_read_only.return_value = ExportMetrics(
        num_rows=0, source_size=0)
    source = MagicMock(connector_id=str(CONNECTOR_ID), table='EMPTY', sql=None)
    destination = str(tmp_path / 'absent')
    with patch(
            'rest.backends.ml_backend.activities.custom_prediction_table.'
            'get_connector_from_spec', return_value=connector), patch(
                'rest.backends.ml_backend.activities.custom_prediction_table.'
                'parquet_util.is_empty') as inspect_files:
        with pytest.raises(EmptyPredictionTableError):
            custom_prediction_table.stage_custom_prediction_table(
                REFERENCE, source, destination)
    inspect_files.assert_not_called()
    connector.close.assert_called_once()
    assert not (tmp_path / 'absent').exists()
