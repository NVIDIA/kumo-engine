# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
from kumoapi.data_source import DataSourceType
from kumoapi.source_table import FileType, S3RelativeSource, S3SourceTable
from kumoapi.typing import Dtype, Stype

from kumoai_studio.connector.base import Connector
from kumoai_studio.connector.source_table import SourceTable
from kumoai_studio.graph import Table


class MockConnector(Connector):
    _kumo_project_id = 'project-test'

    @property
    def id(self) -> str:
        return 'connector-test'

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.S3

    @property
    def name(self) -> str:
        return 'test'


def test_table_basic():
    r"""Test basic table creation flow."""
    source_table = SourceTable('mock_source_table', MockConnector())
    table = Table(
        source_table=source_table,
        columns=[('a', Stype.numerical, Dtype.int), 'b'],
        primary_key=('primary_key', Stype.ID, Dtype.int),
        time_column='a',
    )

    assert str(table) == ('Table(\n'
                          '  source_name=mock_source_table,\n'
                          '  data_source=test,\n'
                          '  columns=[a, b, primary_key],\n'
                          '  primary_key=primary_key,\n'
                          '  time_column=a,\n'
                          '  end_time_column=None,\n'
                          ')')

    assert table.has_primary_key()
    assert table.has_time_column()
    assert not table.has_end_time_column()

    assert table.primary_key is not None
    assert table.primary_key.dtype == Dtype.int
    assert table.time_column is not None
    assert table.time_column.dtype == Dtype.int
    assert table.column('b').dtype is None

    table.time_column = None
    assert not table.has_time_column()
    assert table.has_column('b')
    table.remove_column('b')
    assert not table.has_column('b')

    metadata = table.metadata
    assert metadata.shape == (2, 7)
    assert metadata['multicategorical_delimiter'].isna().all()
    assert metadata['is_primary_key'].sum() == 1
    assert metadata['is_end_time_column'].sum() == 0
    assert set(metadata['name'].to_list()) == {'a', 'primary_key'}


def test_table_name_is_required_only_when_saving(mock_backend):
    r""":meth:`Table.save` takes the name as a required argument and refuses a
    blank one before any backend call, while :meth:`Table.snapshot` and
    :meth:`Table.get_stats` take no name.
    """
    source_table = SourceTable(
        'mock_source_table',
        MockConnector(),
        _api_source_table=S3SourceTable(
            connector_id='connector-test',
            source=S3RelativeSource(path='mock_source_table'),
            file_type=FileType.PARQUET,
        ),
    )
    table = Table(source_table, columns=[('a', Stype.numerical, Dtype.int)])

    assert (inspect.signature(Table.save).parameters['name'].default
            is inspect.Parameter.empty)
    for blank in ('', '   '):
        with pytest.raises(ValueError, match='table name is required'):
            table.save(blank)
    mock_backend.table.create_table.assert_not_called()

    mock_backend.table.get_table_if_exists.return_value = None
    mock_backend.table.create_table.return_value = 'table-saved'
    assert table.save('customer-events') == 'table-saved'
    create_call = mock_backend.table.create_table.call_args
    assert create_call.kwargs['name_alias'] == 'customer-events'
    assert create_call.kwargs['force_rename'] is True

    assert 'name' not in inspect.signature(Table.snapshot).parameters
    assert 'name' not in inspect.signature(Table.get_stats).parameters

    mock_backend.table.create_snapshot.return_value = 'snapshot-1@version-1'
    assert table.snapshot(non_blocking=True) == 'snapshot-1@version-1'
    assert table.get_stats() == {}
    snapshot_call = mock_backend.table.create_snapshot.call_args
    assert set(snapshot_call.kwargs) == {
        'table_definition', 'refresh_source', 'project_id'
    }
