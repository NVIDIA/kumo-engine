# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Tests for the ``update_table_timestamp_units`` data-backend activity.

The activity runs inference over a real connector sample and persists the
resolved temporal hint back onto the frozen v2 ``Column.timestamp_format``,
which overloads on type: a ``TimestampUnit`` enum for a UNIT (epoch) column and
a raw strftime string for a FORMAT (encoded-date / string) column.

The activity is a plain (sync) function with no Temporal dependency in its core,
so it is driven directly here. A local ``ParquetConnector`` exercises ambiguous
value inference end to end, while connector metadata mocks isolate native-schema
and explicit-hint short circuits.
"""
import os
from unittest import mock

import pandas as pd
import pytest
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.source_table import BigQuerySourceTable, FileType, S3SourceTable
from kumoapi.table import Dtype as ApiDtype
from kumoapi.table import Stype as ApiStype

from kumo.config.graph import TableMetadataConfig
from kumo.connector.parquet_connector import ParquetConnector
from rest.backends.data_backend.activities.data_backend_activities import (
    DataBackendActivities,
)
from rest.config.table_v2 import (
    Column,
    CustomSQLSourceTable,
    TableDefinition,
    TimestampUnit,
)


def _parquet_connector(tmp_path, df: pd.DataFrame) -> ParquetConnector:
    # get_source_table_name() returns the connector-relative source path
    # ('test_table'), so read_table_sample resolves it to test_table.parquet
    # under root_dir.
    df.to_parquet(os.path.join(str(tmp_path), 'test_table.parquet'))
    return ParquetConnector(name='test', root_dir=str(tmp_path))


def test_update_table_timestamp_units_unit_and_format(tmp_path):
    df = pd.DataFrame({
        'EPOCH_MS': pd.Series(
            [1659304801388, 1659304901388, 1659305001388], dtype='int64'),
        'ENCODED_DATE': pd.Series(
            [20230408, 20240115, 20250301], dtype='int64'),
        'STRING_DATE': pd.Series(
            ['2023-04-08', '2024-01-15', '2025-03-01'], dtype='string'),
    })
    connector = _parquet_connector(tmp_path, df)

    table_def = TableDefinition(
        cols=[
            Column(name='EPOCH_MS', stype=ApiStype.timestamp,
                   dtype=ApiDtype.int64),
            Column(name='ENCODED_DATE', stype=ApiStype.timestamp,
                   dtype=ApiDtype.int64),
            Column(name='STRING_DATE', stype=ApiStype.timestamp,
                   dtype=ApiDtype.string),
        ],
        source_table=S3SourceTable(
            connector_id=f'connector_{"0" * 32}',
            source={'mode': 'relative', 'path': 'test_table'},
            file_type=FileType.PARQUET),
        pkey=None,
        time_col=None,
    )

    with (
        mock.patch.object(
            connector, 'read_table_sample',
            wraps=connector.read_table_sample) as read_table_sample,
        mock.patch(
            'rest.backends.data_backend.activities.data_backend_activities'
            '.get_connector_from_spec', return_value=connector),
    ):
        updated = DataBackendActivities.update_table_timestamp_units(
            table_def, mock.Mock(), None, None)

    by_name = {col.name: col for col in updated.cols}
    assert read_table_sample.call_count == 1
    # UNIT (epoch) column -> TimestampUnit enum.
    assert by_name['EPOCH_MS'].timestamp_format == TimestampUnit('ms')
    assert isinstance(by_name['EPOCH_MS'].timestamp_format, TimestampUnit)
    # FORMAT (encoded-date) column -> raw strftime string.
    assert by_name['ENCODED_DATE'].timestamp_format == '%Y%m%d'
    assert isinstance(by_name['ENCODED_DATE'].timestamp_format, str)
    # Both source string and integer metadata remain ambiguous and therefore
    # share the same sampled frame.
    assert by_name['STRING_DATE'].timestamp_format == '%Y-%m-%d'


def test_native_bigquery_temporal_metadata_avoids_sample():
    connector = mock.Mock()
    connector.get_table_metadata.return_value = [TableMetadataConfig(
        table_name='events',
        source_name='events',
        data_source='test',
        cols=[
            BaseColumnConfig(name='EVENT_DATE', dtype=ApiDtype.date),
            BaseColumnConfig(name='EVENT_TIME', dtype=ApiDtype.time),
        ],
    )]
    table_def = TableDefinition(
        cols=[
            Column(name='EVENT_DATE', stype=ApiStype.timestamp,
                   dtype=ApiDtype.string),
            Column(name='EVENT_TIME', stype=ApiStype.timestamp,
                   dtype=ApiDtype.int64),
        ],
        source_table=BigQuerySourceTable(
            bigquery_connector_id=f'connector_{"0" * 32}',
            table_name='events',
            project_id='test-project',
            dataset_id='test_dataset'),
        pkey=None,
        time_col=None,
    )

    with mock.patch(
            'rest.backends.data_backend.activities.data_backend_activities'
            '.get_connector_from_spec', return_value=connector):
        updated = DataBackendActivities.update_table_timestamp_units(
            table_def, mock.Mock(), None, None)

    connector.get_table_metadata.assert_called_once_with(
        ['events'], skip_cache=True, skip_inferral=True)
    connector.read_table_sample.assert_not_called()
    assert updated == table_def


@pytest.mark.parametrize('metadata_result', [RuntimeError('provider-secret'),
                                             []])
def test_physical_metadata_unavailable_falls_back_to_sample(
        tmp_path, caplog, metadata_result):
    connector = _parquet_connector(
        tmp_path,
        pd.DataFrame({
            'EPOCH_MS': pd.Series(
                [1659304801388, 1659304901388, 1659305001388], dtype='int64'),
        }),
    )
    table_def = TableDefinition(
        cols=[Column(name='EPOCH_MS', stype=ApiStype.timestamp,
                     dtype=ApiDtype.int64)],
        source_table=S3SourceTable(
            connector_id=f'connector_{"0" * 32}',
            source={'mode': 'relative', 'path': 'test_table'},
            file_type=FileType.PARQUET),
        pkey=None,
        time_col=None,
    )
    metadata_kwargs = (
        {'side_effect': metadata_result}
        if isinstance(metadata_result, Exception)
        else {'return_value': metadata_result})

    with (
        mock.patch.object(connector, 'get_table_metadata', **metadata_kwargs),
        mock.patch.object(
            connector, 'read_table_sample',
            wraps=connector.read_table_sample) as read_table_sample,
        mock.patch(
            'rest.backends.data_backend.activities.data_backend_activities'
            '.get_connector_from_spec', return_value=connector),
    ):
        updated = DataBackendActivities.update_table_timestamp_units(
            table_def, mock.Mock(), None, None)

    assert read_table_sample.call_count == 1
    assert updated.cols[0].timestamp_format == TimestampUnit('ms')
    assert 'falling back to timestamp value inference' in caplog.text
    assert 'provider-secret' not in caplog.text


def test_native_custom_sql_temporal_metadata_avoids_sample():
    sql = 'SELECT event_time FROM events'
    connector = mock.Mock()
    connector.get_query_metadata.return_value = TableMetadataConfig(
        table_name='',
        source_name='',
        data_source='test',
        cols=[BaseColumnConfig(name='EVENT_TIME', dtype=ApiDtype.time)],
    )
    table_def = TableDefinition(
        cols=[Column(name='EVENT_TIME', stype=ApiStype.timestamp,
                     dtype=ApiDtype.string)],
        source_table=CustomSQLSourceTable(
            connector_id=f'connector_{"0" * 32}',
            connector_type='bigquery',
            sql=sql,
            table='events_query',
            referenced_tables=['events']),
        pkey=None,
        time_col=None,
    )

    with mock.patch(
            'rest.backends.data_backend.activities.data_backend_activities'
            '.get_connector_from_spec', return_value=connector):
        updated = DataBackendActivities.update_table_timestamp_units(
            table_def, mock.Mock(), None, None)

    connector.get_query_metadata.assert_called_once_with(sql)
    connector.get_table_metadata.assert_not_called()
    connector.read_table_sample.assert_not_called()
    assert updated == table_def


def test_explicit_temporal_hints_remain_unchanged():
    connector = mock.Mock()
    table_def = TableDefinition(
        cols=[
            Column(name='EPOCH_MS', stype=ApiStype.timestamp,
                   dtype=ApiDtype.int64,
                   timestamp_format=TimestampUnit('ms')),
            Column(name='STRING_DATE', stype=ApiStype.timestamp,
                   dtype=ApiDtype.string,
                   timestamp_format='%Y-%m-%d'),
        ],
        source_table=BigQuerySourceTable(
            bigquery_connector_id=f'connector_{"0" * 32}',
            table_name='events',
            project_id='test-project',
            dataset_id='test_dataset'),
        pkey=None,
        time_col=None,
    )

    with mock.patch(
            'rest.backends.data_backend.activities.data_backend_activities'
            '.get_connector_from_spec', return_value=connector):
        updated = DataBackendActivities.update_table_timestamp_units(
            table_def, mock.Mock(), None, None)

    connector.get_table_metadata.assert_not_called()
    connector.read_table_sample.assert_not_called()
    assert updated == table_def
