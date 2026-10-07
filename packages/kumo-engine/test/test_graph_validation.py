# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from kumoapi.graph import Edge
from kumoapi.source_table import FileType, S3RelativeSource, S3SourceTable
from kumoapi.table import Column, TableDefinition
from kumoapi.typing import Dtype, Stype

from kumo.engine import FileConnector, Graph, S3Connector, Table
from kumo.engine._portable import PortableGraphError


def _file_tables(root: Path) -> list[Table]:
    root.mkdir()
    (root / 'users.csv').write_text('id,name\n1,Alice\n2,Bob\n')
    (root / 'events.csv').write_text('event_id,user_id\n10,1\n11,2\n')
    connector = FileConnector(root_dir=str(root))
    return [
        Table.from_source_table(connector['users'], primary_key='id'),
        Table.from_source_table(connector['events'], primary_key='event_id'),
    ]


def _s3_tables() -> list[Table]:
    connector = S3Connector(root_dir='s3://schema-only-fixture')

    def table(name: str, columns: list[Column], primary_key: str) -> Table:
        definition = TableDefinition(
            cols=columns,
            source_table=S3SourceTable(
                connector_id=connector.id,
                file_type=FileType.PARQUET,
                source=S3RelativeSource(path=name),
            ),
            pkey=primary_key,
        )
        return Table(connector, name, definition)

    return [
        table('users', [Column('id', Stype.ID, Dtype.int64)], 'id'),
        table(
            'events',
            [
                Column('event_id', Stype.ID, Dtype.int64),
                Column('user_id', Stype.ID, Dtype.int64),
            ],
            'event_id',
        ),
    ]


@pytest.fixture(params=['file', 's3'])
def tables(request, tmp_path) -> list[Table]:
    if request.param == 'file':
        return _file_tables(tmp_path / 'data')
    return _s3_tables()


@pytest.mark.parametrize(
    'edge, missing',
    [
        (Edge('missing', 'user_id', 'users'), "missing source table 'missing'"),
        (Edge('events', 'user_id',
              'missing'), "missing destination table 'missing'"),
    ],
)
def test_graph_validation_rejects_missing_edge_endpoints(tables, edge, missing):
    graph = Graph(tables, [edge])

    with pytest.raises(PortableGraphError) as error:
        graph.validate()

    assert missing in str(error.value)
    assert "available tables: ['events', 'users']" in str(error.value)

    with patch.object(
            Graph, '_snapshot_no_lock',
            side_effect=AssertionError('snapshot materialization started')):
        with pytest.raises(PortableGraphError, match=missing):
            graph.snapshot()


def test_graph_validation_keeps_valid_edges(tables):
    graph = Graph(tables, [Edge('events', 'user_id', 'users')])

    graph.validate()

    portable_tables = {
        table['table_name']: table
        for table in graph._graph_config['tables']
    }
    assert portable_tables['events']['fkeys'][0]['dst_table'] == 'users'


def test_graph_validation_rejects_destination_without_primary_key(tables):
    users, events = tables
    users = replace(users, definition=replace(users.definition, pkey=None))
    graph = Graph([users, events], [Edge('events', 'user_id', 'users')])

    with pytest.raises(PortableGraphError,
                       match=r"destination table users.*primary key"):
        graph.validate()

    with patch.object(
            Graph, '_snapshot_no_lock',
            side_effect=AssertionError('snapshot materialization started')):
        with pytest.raises(PortableGraphError,
                           match=r"destination table users.*primary key"):
            graph.snapshot()


def test_graph_validation_warns_when_self_relation_is_omitted(tables):
    from unittest import mock

    graph = Graph(tables, [Edge('events', 'user_id', 'events')])

    with pytest.warns(
            UserWarning,
            match=r"Self-relation events\.user_id -> events\.event_id .* omitted"
    ) as warnings:
        graph.validate()
        with mock.patch.object(Graph, '_snapshot_no_lock'):
            graph.snapshot()

    assert sum('Self-relation' in str(w.message) for w in warnings) == 1

    events = next(table for table in graph._graph_config['tables']
                  if table['table_name'] == 'events')
    assert events['fkeys'] == []


@pytest.mark.parametrize('original_stype', [Stype.numerical, Stype.categorical])
def test_declared_relation_assigns_id_without_mutating_source(
        tables, original_stype):
    source = tables[1].with_column('user_id', stype=original_stype)
    graph = Graph([tables[0], source], [Edge('events', 'user_id', 'users')])
    graph.validate()
    projected = next(table for table in graph._graph_config['tables']
                     if table['table_name'] == 'events')
    assert projected['fkeys'][0]['stype'] == 'ID'
    assert next(col for col in projected['cols']
                if col['name'] == 'user_id')['stype'] == 'ID'
    assert next(col for col in source.definition.cols
                if col.name == 'user_id').stype == original_stype


@pytest.mark.parametrize('original_stype', [Stype.numerical, Stype.categorical])
def test_omitted_self_relation_preserves_column_type(tables, original_stype):
    source = tables[1].with_column('user_id', stype=original_stype)
    graph = Graph([source], [Edge('events', 'user_id', 'events')])
    with pytest.warns(UserWarning, match='Self-relation .* omitted'):
        graph.validate()
    projected = graph._graph_config['tables'][0]
    assert projected['fkeys'] == []
    assert next(col for col in projected['cols']
                if col['name'] == 'user_id')['stype'] == original_stype
    assert next(col for col in source.definition.cols
                if col.name == 'user_id').stype == original_stype
