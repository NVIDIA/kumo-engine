# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import PropertyMock, patch
from xml.etree import ElementTree

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


@pytest.mark.parametrize('alias, key', [
    ('<script>alert("账户")</script> & customers', 'customer<&"账户'),
    ('PURCHASES_AND_TRANSACTIONS', 'O' * 22),
    ('Ж' * 27, 'Ш' * 22),
])
def test_visualize_exports_escaped_svg_without_source_access(
        tmp_path, alias, key):
    users, events = _s3_tables()
    events = replace(
        events, definition=replace(
            events.definition, cols=[
                events.definition.cols[0],
                Column(key, Stype.ID, Dtype.int64)
            ]))
    graph = Graph({
        'purchases': events,
        alias: users
    }, [Edge('purchases', key, alias)])
    destination = tmp_path / 'relationships.svg'
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      side_effect=AssertionError('source accessed')), \
            patch.object(Table, 'infer_metadata',
                         side_effect=AssertionError('inference started')), \
            patch.object(Graph, 'snapshot',
                         side_effect=AssertionError('snapshot started')):
        svg = graph.visualize(destination)
    assert isinstance(svg, str)
    assert svg._repr_svg_() == svg
    assert destination.read_text(encoding='utf-8') == svg
    root = ElementTree.fromstring(svg)
    assert root.tag == '{http://www.w3.org/2000/svg}svg'
    text = ''.join(root.itertext())
    assert alias in text and key in text
    if alias.isupper():
        visible = [
            element.text
            for element in root.iter('{http://www.w3.org/2000/svg}text')
        ]
        assert alias not in visible and key not in visible
    assert not root.findall('.//{http://www.w3.org/2000/svg}script')
    assert all(not name.lower().startswith('on') for element in root.iter()
               for name in element.attrib)
    assert 's3://schema-only-fixture' not in svg


@pytest.mark.parametrize('count, limit', [(10, 10), (11, 5)])
@pytest.mark.parametrize('show_cols', [False, True])
def test_visualize_bounds_column_detail(count, limit, show_cols):
    users, _ = _s3_tables()
    extras = [
        Column(f'feature_{i:02}', Stype.numerical, Dtype.int64)
        for i in range(12)
    ]
    table = replace(
        users, definition=replace(users.definition,
                                  cols=[*users.definition.cols, *extras]))
    graph = Graph({f'table_{i:02}': table for i in range(count)})
    root = ElementTree.fromstring(graph.visualize(show_cols=show_cols))
    text = ' '.join(
        element.text or ''
        for element in root.iter('{http://www.w3.org/2000/svg}text'))
    visible = limit - 1 if show_cols else 0
    for i in range(12):
        assert (f'feature_{i:02}' in text) == (i < visible)
    assert re.search(rf'\+\s*{12 - visible} (other )?columns', text)


def test_visualize_retains_keys_beyond_column_limit():
    users, events = _s3_tables()
    keys = [Column(f'foreign_{i:02}', Stype.ID, Dtype.int64) for i in range(12)]
    events = replace(
        events, definition=replace(events.definition,
                                   cols=[events.definition.cols[0], *keys]))
    graph = Graph([users, events],
                  [Edge('events', column.name, 'users') for column in keys])
    text = ''.join(ElementTree.fromstring(graph.visualize()).itertext())
    assert all(column.name in text for column in keys)
    assert 'event_id' in text and 'id' in text


def test_visualize_layout_is_deterministic_with_cycles_and_isolates():
    _, table = _s3_tables()
    tables = {name: table for name in ['a', 'b', 'c', 'isolated']}
    edges = [
        Edge('a', 'user_id', 'b'),
        Edge('b', 'user_id', 'c'),
        Edge('c', 'user_id', 'a'),
        Edge('a', 'event_id', 'b')
    ]
    svg = Graph(tables, edges).visualize()
    assert svg == Graph(dict(reversed(list(tables.items()))),
                        list(reversed(edges))).visualize()
    root = ElementTree.fromstring(svg)
    text = ''.join(root.itertext())
    assert all(name in text for name in tables)
    assert all(float(value) > 0 for value in root.attrib['viewBox'].split()[2:])
    titles = {
        element.text
        for element in root.iter('{http://www.w3.org/2000/svg}title')
    }
    assert all(
        f'{edge.src_table}.{edge.fkey} → {edge.dst_table}.event_id' in titles
        for edge in edges)
    paths = root.findall('.//{http://www.w3.org/2000/svg}path[@fill="none"]')
    assert len(paths) == len(edges)
    assert len({path.attrib['d'] for path in paths}) == len(edges)
    assert any(re.search('[CQ]', path.attrib['d']) for path in paths)


@pytest.mark.parametrize('case, message', [
    ('unresolved', 'definition|infer_metadata'),
    ('endpoint', 'missing'),
    ('no_primary_key', 'primary key'),
    ('missing_primary_key', 'missing|primary key'),
    ('missing_foreign_key', 'missing|foreign key'),
    ('self_relation', '[Ss]elf'),
    ('duplicate_columns', 'duplicate column'),
    ('xml_control', 'valid XML'),
])
def test_visualize_rejects_invalid_metadata(case, message):
    users, events = _s3_tables()
    edge = Edge('events', 'user_id', 'users')
    if case == 'unresolved':
        users = replace(users, definition=None)
    elif case in ('no_primary_key', 'missing_primary_key'):
        users = replace(
            users, definition=replace(
                users.definition,
                pkey=None if case == 'no_primary_key' else 'gone'))
    elif case == 'endpoint':
        edge = Edge('missing', 'user_id', 'users')
    elif case == 'missing_foreign_key':
        edge = Edge('events', 'gone', 'users')
    elif case == 'self_relation':
        edge = Edge('events', 'user_id', 'events')
    elif case == 'duplicate_columns':
        users = replace(
            users, definition=replace(users.definition,
                                      cols=users.definition.cols * 2))
    elif case == 'xml_control':
        users = replace(users, table_name='users\x00')
        edge = Edge('events', 'user_id', users.table_name)
    with pytest.raises(ValueError, match=message):
        Graph([users, events], [edge]).visualize()


def test_visualize_empty_graph():
    svg = Graph([]).visualize()
    assert 'No tables' in ''.join(ElementTree.fromstring(svg).itertext())
