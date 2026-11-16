# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from kumoapi.data_source import (
    ADLSAnyURIConnectorTarget,
    ADLSRootedConnectorTarget,
    GCSAnyURIConnectorTarget,
    GCSRootedConnectorTarget,
    S3AnyURIConnectorTarget,
    S3RootedConnectorTarget,
)
from kumoapi.jobs import (
    ADLSPredictionTableSource,
    DatabricksPredictionTableSource,
    GCSPredictionTableSource,
    S3PredictionTableSource,
)
from kumoapi.source_table import (
    ADLSRelativeSource,
    ADLSSourceTable,
    ADLSURISource,
    CustomSQLSourceTable,
    DatabricksSourceTable,
    FileType,
    GCSRelativeSource,
    GCSSourceTable,
    GCSURISource,
    S3RelativeSource,
    S3SourceTable,
    S3URISource,
)

from kumoai_studio import (
    ADLSConnector,
    BigQueryConnector,
    DatabricksConnector,
    GCSConnector,
    S3Connector,
    SourceTable,
)
from kumoai_studio.client.endpoints import SourceTableEndpoints
from kumoai_studio.client.source_table import SourceTableAPI
from kumoai_studio.pquery.prediction_table import PredictionTable


def _s3_connector(*, any_uri: bool = False) -> S3Connector:
    connector = object.__new__(S3Connector)
    connector._id = 'connector-s3'
    connector._kumo_project_id = 'project-id'
    connector.any_uri = any_uri
    connector.target = (S3AnyURIConnectorTarget()
                        if any_uri else S3RootedConnectorTarget(
                            bucket='bucket', prefix='root'))
    connector.bucket = None if any_uri else 'bucket'
    connector.prefix = None if any_uri else 'root'
    connector.root_dir = None if any_uri else 's3://bucket/root'
    return connector


def _source_table(connector: S3Connector, name: str, source) -> SourceTable:
    table = SourceTable(name, connector)
    cast(Any,
         table)._to_api_source_table = MagicMock(return_value=S3SourceTable(
             connector_id=connector.id,
             source=source,
             file_type=FileType.PARQUET,
         ))
    return table


def _assert_custom_source_urls(table, connector, name, expected_url):
    state = MagicMock()
    state.client.source_table_api._get_download_urls.return_value = [
        'https://signed.example/part.parquet'
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr('kumoai_studio.global_state', state)
        assert table.data_urls() == [expected_url]
        assert table._data_urls_for_read() == [
            'https://signed.example/part.parquet'
        ]
    state.client.source_table_api._get_download_urls.assert_called_once_with(
        connector.id, name, project_id='project-id')


def test_custom_source_download_transport_is_not_public_sdk_api():
    assert not hasattr(SourceTableEndpoints, 'get_download_urls')
    assert not hasattr(SourceTableAPI, 'get_download_urls')


def test_custom_prediction_table_preserves_deep_rooted_source():
    connector = _s3_connector()
    source = _source_table(
        connector,
        'deep/many/layers/predictions.parquet',
        S3RelativeSource(path='deep/many/layers/predictions.parquet'),
    )

    table = PredictionTable(source_table=source)

    assert table.kumo_project_id == 'project-id'
    assert table._to_api_custom_source() == S3PredictionTableSource(
        connector_id='connector-s3',
        source=S3RelativeSource(path='deep/many/layers/predictions.parquet'),
    )
    _assert_custom_source_urls(
        table, connector, source.name,
        's3://bucket/root/deep/many/layers/predictions.parquet')


def test_rooted_s3_prediction_dataset_preserves_trailing_slash():
    connector = _s3_connector()
    source = _source_table(connector, 'deep/dataset/',
                           S3RelativeSource(path='deep/dataset/'))

    table = PredictionTable(source_table=source)

    _assert_custom_source_urls(table, connector, source.name,
                               's3://bucket/root/deep/dataset/')


def test_custom_prediction_table_preserves_any_uri_source():
    connector = _s3_connector(any_uri=True)
    uri = 's3://other-bucket/deep/many/layers/predictions.parquet'
    source = _source_table(connector, uri, S3URISource(uri=uri))

    table = PredictionTable(source_table=source)

    assert table._to_api_custom_source() == S3PredictionTableSource(
        connector_id='connector-s3', source=S3URISource(uri=uri))
    _assert_custom_source_urls(table, connector, uri, uri)


@pytest.mark.parametrize(
    ('connector', 'name', 'api_source', 'wire_type', 'expected_url'),
    [
        (
            GCSConnector,
            'deep/predictions.parquet',
            GCSRelativeSource(path='deep/predictions.parquet'),
            GCSPredictionTableSource,
            'gs://bucket/root/deep/predictions.parquet',
        ),
        (
            GCSConnector,
            'deep/dataset/',
            GCSRelativeSource(path='deep/dataset/'),
            GCSPredictionTableSource,
            'gs://bucket/root/deep/dataset/',
        ),
        (
            GCSConnector,
            'gs://other-bucket/deep/predictions.parquet',
            GCSURISource(uri='gs://other-bucket/deep/predictions.parquet'),
            GCSPredictionTableSource,
            'gs://other-bucket/deep/predictions.parquet',
        ),
        (
            ADLSConnector,
            'deep/predictions.parquet',
            ADLSRelativeSource(path='deep/predictions.parquet'),
            ADLSPredictionTableSource,
            ('abfss://curated@lake.dfs.core.windows.net/root/'
             'deep/predictions.parquet'),
        ),
        (
            ADLSConnector,
            'deep/dataset/',
            ADLSRelativeSource(path='deep/dataset/'),
            ADLSPredictionTableSource,
            ('abfss://curated@lake.dfs.core.windows.net/root/'
             'deep/dataset/'),
        ),
        (
            ADLSConnector,
            ('abfss://curated@otherlake.dfs.core.chinacloudapi.cn/'
             'deep/predictions.parquet'),
            ADLSURISource(
                uri=('abfss://curated@otherlake.dfs.core.chinacloudapi.cn/'
                     'deep/predictions.parquet')),
            ADLSPredictionTableSource,
            ('abfss://curated@otherlake.dfs.core.chinacloudapi.cn/'
             'deep/predictions.parquet'),
        ),
    ],
)
def test_gcs_adls_custom_prediction_table_reuses_source_coordinate(
        connector, name, api_source, wire_type, expected_url):
    instance = object.__new__(connector)
    instance._id = f'connector-{connector.__name__.lower()}'
    instance._kumo_project_id = 'project-id'
    instance.any_uri = isinstance(api_source, (GCSURISource, ADLSURISource))
    if connector is GCSConnector:
        instance.target = (GCSAnyURIConnectorTarget()
                           if instance.any_uri else GCSRootedConnectorTarget(
                               bucket='bucket', prefix='root'))
        instance.root_dir = None if instance.any_uri else 'gs://bucket/root'
        api_table = GCSSourceTable(
            connector_id=instance.id,
            source=api_source,
            file_type=FileType.PARQUET,
        )
    else:
        instance.target = (ADLSAnyURIConnectorTarget()
                           if instance.any_uri else ADLSRootedConnectorTarget(
                               storage_account='lake', filesystem='curated',
                               prefix='root'))
        instance.root_dir = (None if instance.any_uri else
                             'abfss://curated@lake.dfs.core.windows.net/root')
        api_table = ADLSSourceTable(
            connector_id=instance.id,
            source=api_source,
            file_type=FileType.PARQUET,
        )
    source = SourceTable(name, instance)
    cast(Any, source)._to_api_source_table = MagicMock(return_value=api_table)

    table = PredictionTable(source_table=source)

    wire = table._to_api_custom_source()
    assert isinstance(wire, wire_type)
    assert wire.source == api_source
    _assert_custom_source_urls(table, instance, name, expected_url)


def _databricks_connector() -> DatabricksConnector:
    connector = object.__new__(DatabricksConnector)
    connector._id = 'connector-databricks'
    connector._kumo_project_id = 'project-id'
    return connector


def test_databricks_custom_prediction_table_uses_physical_table():
    connector = _databricks_connector()
    source = SourceTable(
        'prediction_entities',
        connector,
        _api_source_table=DatabricksSourceTable(
            databricks_connector_id=connector.id,
            table='prediction_entities',
        ),
    )

    table = PredictionTable(source_table=source)

    assert table._to_api_custom_source() == DatabricksPredictionTableSource(
        connector_id=connector.id,
        table='prediction_entities',
    )
    with pytest.raises(ValueError, match='do not expose downloadable'):
        table.data_urls()


def test_databricks_custom_prediction_table_uses_read_only_sql():
    connector = _databricks_connector()
    source = connector.sql(
        'SELECT entity_id, event_time AS TIMESTAMP FROM prediction_entities')
    cast(Any, source)._to_api_source_table = MagicMock(
        return_value=CustomSQLSourceTable(
            connector_id=connector.id,
            sql=source.sql,
            table='custom_sql_prediction',
        ))

    table = PredictionTable(source_table=source)

    assert table._to_api_custom_source() == DatabricksPredictionTableSource(
        connector_id=connector.id,
        sql=source.sql,
    )
    with pytest.raises(ValueError, match='cannot be downloaded'):
        table.data_df()


def test_custom_prediction_table_rejects_non_object_store_source():
    connector = object.__new__(BigQueryConnector)
    connector._kumo_project_id = 'project-id'
    source = SourceTable('deep/table.parquet', connector)

    with pytest.raises(ValueError, match='ADLSConnector.*DatabricksConnector'):
        PredictionTable(source_table=source)


def test_custom_prediction_table_rejects_cross_project():
    source = _source_table(
        _s3_connector(),
        'deep/table.parquet',
        S3RelativeSource(path='deep/table.parquet'),
    )

    with pytest.raises(ValueError, match='belongs to project'):
        PredictionTable(source_table=source, project_id='other-project')


def test_raw_custom_prediction_path_is_not_public_api():
    with pytest.raises(TypeError, match='table_data_path'):
        cast(Any, PredictionTable)(table_data_path='s3://bucket/table.parquet')


def test_generated_prediction_table_uses_server_selected_safe_references(
        mock_backend):
    table = object.__new__(PredictionTable)
    table.job_id = 'generate-prediction-table-job'
    table.source_table = None
    table._kumo_project_id = 'project-owned'
    data_api = mock_backend.prediction_table
    data_api.get_table_data.return_value = [
        'https://safe-reference/part-0', 'https://safe-reference/part-1'
    ]

    assert table.data_urls() == [
        'https://safe-reference/part-0',
        'https://safe-reference/part-1',
    ]

    data_api.get_table_data.assert_called_once_with(
        'generate-prediction-table-job', project_id='project-owned')
