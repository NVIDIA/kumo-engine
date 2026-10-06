# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import MagicMock, patch

import kumoapi.graph as graph_api
import kumoapi.table as table_api
import pytest
from kumoapi.data_source import DataSourceType
from kumoapi.source_table import (
    FileType,
    S3RelativeSource,
    S3SourceTable,
    SourceColumn,
    SourceTableConfigResponse,
)
from kumoapi.typing import Dtype, Stype

from kumoai_studio.connector.base import Connector
from kumoai_studio.connector.source_table import SourceTable
from kumoai_studio.graph import Column, Graph, Table


class _SlowConnector(Connector):
    def __init__(self, *, project_id='project-a'):
        self._kumo_project_id = project_id
        self.calls = 0

    @property
    def id(self):
        return 'connector-slow'

    @property
    def name(self):
        return 'slow'

    @property
    def source_type(self):
        return DataSourceType.S3

    def _get_table_config(self, table_name):
        self.calls += 1
        time.sleep(0.01)
        return SourceTableConfigResponse(source_table=S3SourceTable(
            connector_id=self.id,
            source=S3RelativeSource(path=table_name),
            file_type=FileType.PARQUET,
        ))


def _table(connector, name):
    return Table(
        SourceTable(name, connector),
        columns=[('id', Stype.ID, Dtype.int64)],
        primary_key='id',
    )


def _successful_validation():
    response = MagicMock(ok=True)
    response.empty.return_value = True
    return response


def _s3_relative_path(source_table):
    assert isinstance(source_table, S3SourceTable)
    assert isinstance(source_table.source, S3RelativeSource)
    return source_table.source.path


def test_source_coordinate_is_resolved_once_under_concurrent_serialization():
    connector = _SlowConnector()
    source = SourceTable('nested/many/layers/events.parquet', connector)

    with ThreadPoolExecutor(max_workers=8) as executor:
        resolved = list(
            executor.map(lambda _: source._to_api_source_table(), range(16)))

    assert connector.calls == 1
    assert all(value == resolved[0] for value in resolved)
    assert _s3_relative_path(
        resolved[0]) == 'nested/many/layers/events.parquet'


def test_connector_table_reuses_the_coordinate_resolved_for_existence():
    connector = _SlowConnector()

    source = connector.table('nested/events.parquet')

    assert connector.calls == 1
    assert _s3_relative_path(
        source._to_api_source_table()) == 'nested/events.parquet'
    assert connector.calls == 1


def test_refresh_source_coordinate_is_explicit_and_invalidates_once():
    connector = _SlowConnector()
    source = SourceTable('events.parquet', connector)

    assert _s3_relative_path(source._to_api_source_table()) == 'events.parquet'
    assert _s3_relative_path(source._to_api_source_table()) == 'events.parquet'
    assert connector.calls == 1

    assert source.refresh_source_coordinate() is source
    assert connector.calls == 2
    assert _s3_relative_path(source._to_api_source_table()) == 'events.parquet'
    assert connector.calls == 2


def test_resolved_source_coordinate_survives_deepcopy_without_refetch():
    connector = _SlowConnector()
    source = SourceTable('events.parquet', connector)
    source._to_api_source_table()

    copied = deepcopy(source)

    assert _s3_relative_path(copied._to_api_source_table()) == 'events.parquet'
    assert connector.calls == 1
    assert _s3_relative_path(source._to_api_source_table()) == 'events.parquet'
    assert connector.calls == 1


def test_seven_table_graph_serialization_fetches_each_coordinate_once():
    connector = _SlowConnector()
    graph = Graph({
        f'table_{index}': _table(connector, f'table_{index}')
        for index in range(7)
    })

    definitions = [graph._to_api_graph_definition() for _ in range(8)]

    assert connector.calls == 7
    assert all(definition == definitions[0] for definition in definitions)
    assert graph.kumo_project_id == 'project-a'
    source_tables = [
        table.source_table for table in definitions[0].tables.values()
    ]
    assert all(
        isinstance(source_table, S3SourceTable)
        for source_table in source_tables)
    assert {
        source_table.connector_id
        for source_table in source_tables
        if isinstance(source_table, S3SourceTable)
    } == {'connector-slow'}


def test_graph_serialization_groups_edges_by_destination_table():
    connector = _SlowConnector()
    users = _table(connector, 'users')
    conversions = Table(
        SourceTable('conversions', connector),
        columns=[('id', Stype.ID, Dtype.int64),
                 ('user_id', Stype.ID, Dtype.int64)],
        primary_key='id',
    )
    video_views = Table(
        SourceTable('video_views', connector),
        columns=[('id', Stype.ID, Dtype.int64),
                 ('user_id', Stype.ID, Dtype.int64)],
        primary_key='id',
    )
    graph = Graph({
        'users': users,
        'conversions': conversions,
        'video_views': video_views,
    })
    graph.link('conversions', 'user_id', 'users')
    graph.link('video_views', 'user_id', 'users')

    definition = graph._to_api_graph_definition()

    assert len(definition.col_groups) == 1
    assert {(key.table_name, key.col_name)
            for key in definition.col_groups[0].columns} == {
                ('users', 'id'),
                ('conversions', 'user_id'),
                ('video_views', 'user_id'),
            }


@pytest.mark.parametrize(
    'edge, code',
    [
        (graph_api.Edge('missing', 'user_id',
                        'users'), 'missing_source_table'),
        (graph_api.Edge('events', 'user_id',
                        'missing'), 'missing_destination_table'),
    ],
)
def test_graph_serialization_rejects_edges_mutated_around_link(edge, code):
    connector = _SlowConnector()
    graph = Graph({
        'users': _table(connector, 'users'),
        'events': _table(connector, 'events'),
    })
    graph.edges.append(edge)

    with pytest.raises(graph_api.GraphSpecError) as error:
        graph._to_api_graph_definition()

    assert error.value.code == code
    assert error.value.edge == edge
    assert "available tables: ['events', 'users']" in str(error.value)


def test_graph_deserialization_preserves_a_self_referential_edge():
    connector = _SlowConnector()
    users = Table(
        SourceTable('users', connector),
        columns=[('id', Stype.ID, Dtype.int64),
                 ('legacy_id', Stype.ID, Dtype.int64)],
        primary_key='id',
    )
    events = Table(
        SourceTable('events', connector),
        columns=[('user_id', Stype.ID, Dtype.int64)],
    )
    definition = graph_api.GraphDefinition(
        tables={
            'users': users._to_api_table_definition(),
            'events': events._to_api_table_definition(),
        },
        col_groups=[
            graph_api.ColumnKeyGroup((
                graph_api.ColumnKey('users', 'id'),
                graph_api.ColumnKey('users', 'legacy_id'),
                graph_api.ColumnKey('events', 'user_id'),
            ))
        ],
    )

    edges = Graph._edges_from_api_graph_definition(definition)
    assert [(edge.src_table, edge.fkey, edge.dst_table) for edge in edges] == [
        ('users', 'legacy_id', 'users'),
        ('events', 'user_id', 'users'),
    ]


def test_graph_validate_is_one_canonical_remote_phase(mock_backend):
    connector = _SlowConnector()
    graph = Graph({
        f'table_{index}': _table(connector, f'table_{index}')
        for index in range(7)
    })
    mock_backend.graph.validate_graph.return_value = _successful_validation()

    assert graph.validate(verbose=False) is graph

    assert connector.calls == 7
    mock_backend.graph.validate_graph.assert_called_once()
    mock_backend.table.validate_table.assert_not_called()


def test_multicategorical_delimiter_survives_table_definition_round_trip():
    connector = _SlowConnector()
    table = Table(
        SourceTable('events', connector),
        columns=[
            Column(
                'interests',
                stype='multicategorical',
                dtype='string',
                multicategorical_delimiter='|',
            )
        ],
    )

    definition = table._to_api_table_definition()
    assert definition.cols[0].multicategorical_delimiter == '|'

    with patch.object(
            SourceTable,
            '_from_api_table_definition',
            return_value=table.source_table,
    ):
        loaded = Table._from_api_table_definition(definition)

    assert loaded.column('interests').multicategorical_delimiter == '|'
    assert loaded._to_api_table_definition() == definition


def test_infer_metadata_sends_and_merges_multicategorical_delimiter(
        mock_backend):
    connector = _SlowConnector()
    table = Table(
        SourceTable('events', connector),
        columns=[
            Column(
                'interests',
                stype='multicategorical',
                dtype='string',
                multicategorical_delimiter=';',
            )
        ],
    )
    mock_backend.table.infer_metadata.return_value = (
        table_api.TableMetadataResponse(
            cols=[
                table_api.Column(
                    name='interests',
                    stype=Stype.multicategorical,
                    dtype=Dtype.string,
                    multicategorical_delimiter=';',
                )
            ],
            source_table=table.source_table._to_api_source_table(),
        ))

    assert table.infer_metadata() is table

    request = mock_backend.table.infer_metadata.call_args.args[0]
    assert request.cols[0].stype == Stype.multicategorical
    assert request.cols[0].dtype == Dtype.string
    assert request.cols[0].multicategorical_delimiter == ';'
    assert table.column('interests').multicategorical_delimiter == ';'


class _StaticColumnsSource(SourceTable):
    def __init__(self, name, connector, columns):
        super().__init__(name, connector)
        self._columns = columns

    @property
    def column_dict(self):
        return {column.name: column for column in self._columns}


def test_source_column_delimiter_is_copied_by_both_table_constructors():
    connector = _SlowConnector()
    source_column = SourceColumn(
        name='interests',
        stype=Stype.multicategorical,
        dtype=Dtype.string,
        is_primary=False,
        multicategorical_delimiter='\t',
    )
    source = _StaticColumnsSource('events', connector, [source_column])

    direct = Table(source, columns=[source_column])
    inferred = Table.from_source_table(source)

    assert direct.column('interests').multicategorical_delimiter == '\t'
    assert inferred.column('interests').multicategorical_delimiter == '\t'


@pytest.mark.parametrize('skip_validation', [False, True])
def test_graph_save_relies_on_one_server_create_phase(skip_validation,
                                                      mock_backend):
    connector = _SlowConnector()
    graph = Graph({'events': _table(connector, 'events')})
    mock_backend.graph.create_graph.return_value = 'graph-id'

    assert graph.save(skip_validation=skip_validation) == 'graph-id'

    assert connector.calls == 1
    mock_backend.graph.validate_graph.assert_not_called()
    mock_backend.table.validate_table.assert_not_called()
    mock_backend.graph.create_graph.assert_called_once()
    assert mock_backend.graph.create_graph.call_args.kwargs[
        'skip_validation'] is skip_validation
