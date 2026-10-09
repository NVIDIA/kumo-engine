# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Generator

import pytest
from kumoapi.data_source import ConnectorType, DataSourceType
from kumoapi.graph import GraphDefinition
from kumoapi.source_table import (
    CustomSQLSourceTable,
    FileType,
    S3SourceTable,
    S3URISource,
)
from kumoapi.table import Column as ApiColumn
from kumoapi.table import TableDefinition
from kumoapi.typing import Dtype, Stype

from kumoai_studio.connector import FileUploadConnector, SourceTable
from kumoai_studio.connector.base import Connector
from kumoai_studio.connector.source_table import SQLSourceTable
from kumoai_studio.graph import Edge, Graph, Table
from kumoai_studio.graph.graph import _source_key


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


class MockBigQueryConnector(MockConnector):
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.BIGQUERY


def test_graph_from_definition_rebinds_custom_sql_to_the_new_connector():
    source = CustomSQLSourceTable(
        connector_id='old-process-connector',
        sql='SELECT user_id FROM production.users',
        table='custom_sql_canonical',
        connector_type=ConnectorType.BIGQUERY,
        referenced_tables=['production.users'],
    )
    definition = GraphDefinition(
        tables={
            'users':
            TableDefinition(
                cols=[ApiColumn('user_id', Stype.ID, Dtype.string)],
                source_table=source,
                pkey='user_id',
            ),
        },
        col_groups=[],
    )
    connector = MockBigQueryConnector()

    graph = Graph.from_definition(definition, connector)

    rebound = graph['users'].source_table
    assert isinstance(rebound, SQLSourceTable)
    assert rebound.connector is connector
    assert rebound.name == source.table
    assert rebound.sql == source.sql
    assert rebound.connector_type == source.connector_type
    assert rebound.referenced_tables == source.referenced_tables
    assert rebound._to_api_source_table().connector_id == connector.id


def test_source_key_preserves_a_multi_dot_object_store_uri():
    uri = ('s3://bucket/data/part-00094-tid-1-uuid-1660-1.'
           'c000.snappy.parquet')
    source = S3SourceTable(connector_id='connector-s3',
                           file_type=FileType.PARQUET,
                           source=S3URISource(uri=uri))

    assert source.table.endswith('c000.snappy')
    assert _source_key('EVENTS', source) == uri


def test_graph_basic():
    r"""Test basic graph creation flow."""
    # Setup:
    source_a = SourceTable('a', MockConnector())
    source_b = SourceTable('b', MockConnector())
    source_c = SourceTable('c', MockConnector())

    table_a = Table(
        source_table=source_a,
        columns=[('c1', Stype.numerical, Dtype.int), 'c2'],
        primary_key=('c1', Stype.ID, Dtype.int),
        time_column='c2',
    )
    table_b = Table(
        source_table=source_b,
        columns=[('c1', Stype.text, Dtype.string), 'c2', 'c3'],
        time_column='c2',
        end_time_column='c3',
    )
    table_c = Table(
        source_table=source_c,
        columns=[('c1', Stype.text, Dtype.string),
                 ('c2', Stype.numerical, Dtype.int), 'c3'],
    )

    # Creation:
    graph = Graph({'a': table_a, 'b': table_b, 'c': table_c})

    # Indexing:
    assert isinstance(graph['a'], Table)
    with pytest.raises(KeyError):
        _ = graph['d']

    # Contains:
    assert 'a' in graph
    assert 'd' not in graph

    # Edges:
    graph.link('b', 'c2', 'a')
    with pytest.raises(ValueError):
        graph.link(None)
    with pytest.raises(ValueError):
        graph.link('b', 'c2', 'a')
    with pytest.raises(ValueError):
        graph.link(Edge('b', 'c2', 'a'))

    assert str(graph) == (
        "Graph(\n"
        "  tables=[a, b, c],\n"
        "  edges=[Edge(src_table='b', fkey='c2', dst_table='a')],\n"
        ")")


@pytest.mark.skip(reason="graph hashing is not implemented")
def test_graph_hash():
    r"""Test that graph hashed by its edges and tables."""
    source_a = SourceTable('a', MockConnector())
    source_b = SourceTable('b', MockConnector())
    graph1 = Graph()
    graph1.add_table(
        'a',
        Table(
            source_table=source_a,
            columns=['c1', 'c2'],
            primary_key='c1',
        ))
    graph1.add_table(
        'b',
        Table(
            source_table=source_b,
            columns=['c1', 'c2'],
            primary_key='c2',
        ))
    graph1.link('a', 'c2', 'b')

    graph2 = Graph()
    graph2.add_table(
        'a',
        Table(
            source_table=source_a,
            columns=['c1', 'c2'],
            primary_key='c1',
        ))
    graph2.add_table(
        'b',
        Table(
            source_table=source_b,
            columns=['c1', 'c2'],
            primary_key='c2',
        ))
    graph2.link('a', 'c2', 'b')

    assert hash(graph1) == hash(graph2)


def test_graph_rejects_tables_from_another_project_before_request():
    connector_a = MockConnector()
    connector_a._kumo_project_id = 'project-a'
    connector_b = MockConnector()
    connector_b._kumo_project_id = 'project-b'
    table_a = Table(SourceTable('a', connector_a), columns=[])
    table_b = Table(SourceTable('b', connector_b), columns=[])

    graph = Graph({'a': table_a})
    with pytest.raises(ValueError, match='project-b.*project-a'):
        graph.add_table('b', table_b)

    assert graph.tables == {'a': table_a}


@pytest.fixture
def integ_source_table(setup_integration_client,
                       tmp_path) -> Generator[SourceTable, None, None]:
    connector = FileUploadConnector(file_type='csv')
    source_name = f'airlines_{uuid.uuid4().hex}'
    source_path = tmp_path / f'{source_name}.csv'
    source_path.write_text(
        'IATA_CODE,AIRLINE\nUA,United Airlines\nDL,Delta Air Lines\n',
        encoding='utf-8',
    )
    connector.upload(source_name, str(source_path), auto_partition=False)
    try:
        yield connector[source_name]
    finally:
        connector.delete(source_name)


@pytest.mark.integration
def test_graph_persistence(integ_source_table: SourceTable):
    # TODO(manan): test is not idempotent. Fix by adding a finalizer for
    # cleanup
    airlines = Table(
        integ_source_table,
        columns=['IATA_CODE', 'AIRLINE'],
        primary_key='IATA_CODE',
    )
    airlines['IATA_CODE'].stype = Stype.ID
    airlines['IATA_CODE'].dtype = Dtype.string
    airlines['AIRLINE'].stype = Stype.text
    airlines['AIRLINE'].dtype = Dtype.string

    graph = Graph({'airlines': airlines})
    graph.save()
