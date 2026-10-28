# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch
from xml.etree import ElementTree

import pytest
from kumoapi.graph import Edge
from kumoapi.source_table import (
    CustomSQLSourceTable,
    FileType,
    S3RelativeSource,
    S3SourceTable,
)
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
    ('sales  data', 'order  id'),
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
    html = ElementTree.fromstring(svg._repr_html_())
    assert html.get('data-highlight') == 'row'
    shaded = ElementTree.fromstring(svg._repr_html_(_highlight_table=True))
    assert shaded.get('data-highlight') == 'table'
    embedded = html.find('.//{http://www.w3.org/2000/svg}svg')
    assert re.sub(rb'kumo-[0-9a-f]{32}', b'kumo-ID',
                  ElementTree.tostring(embedded)) == re.sub(
                      rb'kumo-[0-9a-f]{32}', b'kumo-ID',
                      ElementTree.tostring(root))
    assert html.find('.//div[@role="region"]').get('tabindex') == '0'
    text = ''.join(element.text or ''
                   for element in root.iter('{http://www.w3.org/2000/svg}text')
                   if element.get('{http://www.w3.org/XML/1998/namespace}space')
                   == 'preserve')
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
    tips = root.findall('.//{http://www.w3.org/2000/svg}g[@data-kumo-tip]')
    assert len(tips) == 5  # Two headers, three visible key rows.
    assert all(tip.get('tabindex') == '0' for tip in tips)
    details = [tip.get('data-kumo-tip') for tip in tips]
    assert any(f'References: {alias}.id' in tip for tip in details)
    assert any('Role: FK' in tip for tip in details)
    assert any('Type: int64\nSemantic type: ID\nRole: FK' in tip
               for tip in details)
    assert any(f'Connected tables:\n{alias}' in tip for tip in details)
    edge = root.find('.//{http://www.w3.org/2000/svg}path[@class="kumo-edge"]')
    assert edge.get('data-dst-table') == alias
    assert edge.get('data-src-column') == key
    assert 's3://schema-only-fixture' not in svg


@pytest.mark.parametrize('count, limit', [(1, 10), (10, 10), (11, 5)])
@pytest.mark.parametrize('show_cols', [False, True])
@pytest.mark.parametrize('with_time', [False, True])
def test_visualize_bounds_column_detail(count, limit, show_cols, with_time):
    users, _ = _s3_tables()
    extras = [
        Column(f'feature_{i:03}', Stype.numerical, Dtype.int64)
        for i in range(120)
    ]
    table = replace(
        users, definition=replace(users.definition,
                                  time_col='event_time' if with_time else None,
                                  cols=[*users.definition.cols, *extras] + (
                                      [Column('event_time', Stype.timestamp,
                                              Dtype.date)] if with_time else [])))
    graph = Graph({f'table_{i:02}': table for i in range(count)})
    root = ElementTree.fromstring(graph.visualize(show_cols=show_cols))
    text = ' '.join(
        element.text or ''
        for element in root.iter('{http://www.w3.org/2000/svg}text'))
    visible = limit - 1 - int(with_time) if show_cols else 0
    for i in range(120):
        assert (f'feature_{i:03}' in text) == (i < visible)
    assert re.search(rf'\+\s*{120 - visible} (other )?columns', text)
    tips = root.findall('.//{http://www.w3.org/2000/svg}g[@data-kumo-column="event_time"]')
    assert len(tips) == (count if with_time else 0)
    assert all('Role: T' in tip.get('data-kumo-tip') for tip in tips)


@pytest.mark.parametrize('time_column', [None, 'occurred_at', 'event_id', 'foreign_00'])
@pytest.mark.parametrize('show_cols', [False, True])
def test_visualize_retains_keys_beyond_column_limit(time_column, show_cols):
    users, events = _s3_tables()
    keys = [Column(f'foreign_{i:02}', Stype.ID, Dtype.int64) for i in range(12)]
    events = replace(
        events, definition=replace(events.definition,
                                   time_col=time_column,
                                   cols=[events.definition.cols[0], *keys,
                                         Column('occurred_at', Stype.timestamp,
                                                Dtype.date)]))
    edges = [Edge('events', column.name, 'users') for column in keys]
    if time_column == 'event_id':
        edges.append(Edge('events', 'event_id', 'users'))
    graph = Graph([users, events], edges)
    root = ElementTree.fromstring(graph.visualize(show_cols=show_cols))
    text = ''.join(root.itertext())
    assert all(column.name in text for column in keys)
    assert 'event_id' in text and 'id' in text
    if time_column:
        row = root.find('.//{http://www.w3.org/2000/svg}g'
                        f'[@data-kumo-column="{time_column}"]')
        role = {'event_id': 'PK/FK/T', 'foreign_00': 'FK/T'}.get(time_column, 'T')
        assert f'Role: {role}' in row.get('data-kumo-tip')
    else:
        assert root.find('.//{http://www.w3.org/2000/svg}g'
                         '[@data-kumo-column="occurred_at"]') is None


@pytest.mark.parametrize('ambiguous', [False, True])
@pytest.mark.parametrize('layout', ['auto', 'standard', 'compact'])
def test_visualize_layout_is_deterministic_with_cycles_and_isolates(
        ambiguous, layout):
    _, table = _s3_tables()
    tables = {name: table for name in ['a', 'b', 'c', 'isolated']}
    edges = [
        Edge('a', 'user_id', 'b'),
        Edge('b', 'user_id', 'c'),
        Edge('c', 'user_id', 'a'),
        Edge('a', 'event_id', 'b')
    ]
    if ambiguous:
        tables = {
            name:
            replace(
                table, definition=replace(
                    table.definition, pkey=key, cols=[
                        *table.definition.cols,
                        Column(key, Stype.ID, Dtype.int64)
                    ]))
            for name, key in [('a', 'b.c'), ('a.b', 'c')]
        } | {
            'c': table,
            'isolated': table
        }
        edges = [
            Edge('c', 'user_id', 'a.b'),
            Edge('c', 'event_id', 'a'),
            Edge('a.b', 'user_id', 'c'),
            Edge('a', 'event_id', 'c')
        ]
    svg = Graph(tables, edges).visualize(layout=layout)
    other = Graph(dict(reversed(list(tables.items()))),
                  list(reversed(edges))).visualize(layout=layout)
    assert re.sub(r'kumo-[0-9a-f]{32}', 'kumo-ID',
                  svg) == re.sub(r'kumo-[0-9a-f]{32}', 'kumo-ID', other)
    ids = lambda value: set(re.findall(r' id="([^"]+)"', value))
    assert ids(svg).isdisjoint(ids(other))
    root = ElementTree.fromstring(svg)
    text = ''.join(root.itertext())
    assert all(name in text for name in tables)
    assert all(float(value) > 0 for value in root.attrib['viewBox'].split()[2:])
    titles = {
        element.text
        for element in root.iter('{http://www.w3.org/2000/svg}title')
    }
    assert all(f'{edge.src_table}.{edge.fkey} → {edge.dst_table}.'
               f'{tables[edge.dst_table].definition.pkey}' in titles
               for edge in edges)
    paths = root.findall('.//{http://www.w3.org/2000/svg}path[@fill="none"]')
    assert len(paths) == len(edges)
    assert {
        tuple(
            path.get(f'data-{part}')
            for part in ('src-table', 'src-column', 'dst-table', 'dst-column'))
        for path in paths
    } == {(edge.src_table, edge.fkey, edge.dst_table,
           tables[edge.dst_table].definition.pkey)
          for edge in edges}
    assert len({path.attrib['d'] for path in paths}) == len(edges)
    assert any(re.search('[CQ]', path.attrib['d']) for path in paths)


@pytest.mark.parametrize('case, message', [
    ('unresolved', 'definition|infer_metadata'),
    ('endpoint', 'missing'),
    ('no_primary_key', 'primary key'),
    ('missing_primary_key', 'missing|primary key'),
    ('missing_foreign_key', 'missing|foreign key'),
    ('missing_time_column', 'Time column users.gone is missing'),
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
    elif case == 'missing_time_column':
        users = replace(users, definition=replace(users.definition, time_col='gone'))
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


@pytest.mark.parametrize('layout',
                         [None, '', 'flow', 'original', 'Standard', 'circular', [], 1])
def test_visualize_rejects_invalid_layout_before_source_access(layout):
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      side_effect=AssertionError('source accessed')):
        with pytest.raises(ValueError, match='layout must be'):
            Graph(_s3_tables()).visualize(layout=layout, sample_rows=3)


def test_visualize_layouts_preserve_schema_and_export_selection(tmp_path):
    graph = Graph(_s3_tables(), [Edge('events', 'user_id', 'users')])
    schemas = []
    svg_ns = '{http://www.w3.org/2000/svg}'
    for layout in ('standard', 'compact'):
        destination = tmp_path / f'{layout}.svg'
        with patch.object(S3Connector, 'source_backend',
                          new_callable=PropertyMock,
                          side_effect=AssertionError('source accessed')):
            svg = graph.visualize(destination, layout=layout, show_cols=True)
        assert destination.read_text() == svg
        root = ElementTree.fromstring(svg)
        assert not root.findall(f'.//{svg_ns}script')
        cards = root.findall(f'{svg_ns}g[@data-kumo-card]')
        paths = root.findall(f'.//{svg_ns}path[@class="kumo-edge"]')
        schemas.append((
            [card.get('data-kumo-card') for card in cards],
            [''.join(card.itertext()) for card in cards],
            [tuple(path.get(f'data-{part}') for part in
                   ('src-table', 'src-column', 'dst-table', 'dst-column'))
             for path in paths],
        ))
    assert schemas[0] == schemas[1]


@pytest.mark.parametrize('sample_rows', [-1, 4, 1.5, True, None])
def test_visualize_rejects_invalid_sample_count(sample_rows):
    with pytest.raises(ValueError, match='sample_rows'):
        Graph(_s3_tables()).visualize(sample_rows=sample_rows)


@pytest.mark.parametrize(
    'outcome', ['values', 'empty', 'unavailable', 'error', 'sql', 'collision',
                'nulls'])
def test_visualize_samples_once_and_keeps_svg_metadata_only(tmp_path, outcome):
    import pandas as pd

    tables = _s3_tables()
    sql = 'SELECT id FROM users' if outcome == 'sql' else None
    if sql:
        tables[0] = replace(
            tables[0], definition=replace(
                tables[0].definition, source_table=CustomSQLSourceTable(
                    connector_id=tables[0].connector.id, table='users',
                    sql=sql)))
    secret = '<script>alert("private sample")</script>'
    frame = pd.DataFrame({
        'id': [secret, None, ['alpha', 'beta'], 'later non-null'],
        'event_id': ['x' * 1000, None, 'third', 'later non-null'],
        'user_id': [1, 2, 3, 4],
    })
    if outcome == 'nulls':
        frame = pd.DataFrame({
            'id': [None, pd.NA, pd.NaT],
            'event_id': [float('nan')] * 3,
            'user_id': [None, 7, None],
        })
    if outcome == 'empty':
        frame = frame.iloc[:0]
    backend = Mock()
    backend.get_sample.return_value.sample_rows = (
        None if outcome == 'unavailable' else frame.to_json(orient='table'))
    if outcome == 'error':
        backend.get_sample.side_effect = RuntimeError('credential secret')
    if outcome == 'collision':
        tables = [
            replace(
                table, table_name=name,
                definition=replace(table.definition,
                                   cols=[Column(key, Stype.ID,
                                                Dtype.int64)], pkey=key))
            for table, name, key in zip(tables, ['a', 'a.b'], ['b.c', 'c'])
        ]
        backend.get_sample.side_effect = lambda connector, name, *args, **kw: Mock(
            sample_rows=pd.DataFrame({
                'b.c' if name == 'a' else 'c': [name]
            }).to_json(orient='table'))
    destination = tmp_path / 'graph.svg'
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      return_value=backend), patch.object(
                          Graph, 'snapshot', side_effect=AssertionError):
        svg = Graph(tables).visualize(destination, sample_rows=3,
                                      show_cols=True)
        html = svg._repr_html_()
        repeat = svg._repr_html_()
        assert re.sub(r'kumo-[0-9a-f]{32}', 'kumo-ID',
                      html) == re.sub(r'kumo-[0-9a-f]{32}', 'kumo-ID', repeat)
        assert set(re.findall(r' id="([^"]+)"', html)).isdisjoint(
            re.findall(r' id="([^"]+)"', repeat))
    assert backend.get_sample.call_count == 2
    for call, table in zip(backend.get_sample.call_args_list,
                           sorted(tables, key=lambda table: table.table_name)):
        assert call.args == (table.connector, table.table_name, 30)
        assert call.kwargs == {
            'sql': sql if table.table_name == 'users' else None,
            'timeout': 10
        }
    assert 'Sample values' not in svg
    assert svg._repr_svg_() == destination.read_text() == str(svg)
    assert 'credential secret' not in html
    assert '<script>alert' not in html
    tips = ElementTree.fromstring(html).findall(
        './/{http://www.w3.org/2000/svg}g[@data-kumo-tip]')
    details = '\n'.join(tip.get('data-kumo-tip') for tip in tips)
    if outcome in ('values', 'sql'):
        assert secret in details and '\nnull' not in details
        assert "['alpha', 'beta']" in details
        assert 'later non-null' in details
        assert 'x' * 121 not in details
    elif outcome == 'collision':
        rows = [tip for tip in tips if tip.get('data-kumo-row')]
        assert [(tip.get('data-kumo-table'), tip.get('data-kumo-column'))
                for tip in rows] == [('a', 'b.c'), ('a.b', 'c')]
        assert [
            tip.get('data-kumo-tip').split('Sample values:\n')[1]
            for tip in rows
        ] == ["'a'", "'a.b'"]
    elif outcome == 'nulls':
        assert 'No non-null values in preview' in details
        assert 'Sample values:\n7' in details
    elif outcome == 'empty':
        assert 'Sample values: no rows returned' in details
    else:
        assert 'Sample values: unavailable' in details


@pytest.mark.parametrize('count', [1, 2, 3])
def test_visualize_samples_bound_preview_and_select_columns_independently(count):
    import pandas as pd

    from kumo.engine._visualization import _preview

    table = _s3_tables()[0]
    backend = Mock()
    backend.get_sample.return_value.sample_rows = pd.DataFrame({
        'id': [None] * 29 + ['last allowed', 'outside preview'],
        'other': ['first', None, 'second', 'third'] + [None] * 27,
    }).to_json(orient='table')
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      return_value=backend):
        preview = _preview(table, count, ['id', 'other'])
    assert preview['id'] == "\nSample values:\n'last allowed'"
    assert preview['other'].splitlines()[2:] == [
        "'first'", "'second'", "'third'"
    ][:count]
    backend.get_sample.assert_called_once_with(
        table.connector, table.table_name, 30, sql=None, timeout=10)


@pytest.mark.parametrize('count', [30, 31])
def test_visualize_table_cap_precedes_source_access(count):
    table = _s3_tables()[0]
    graph = Graph({f'table_{i:02}': table for i in range(count)})
    with patch.object(S3Connector, 'source_backend',
                      new_callable=PropertyMock) as backend:
        if count == 31:
            with pytest.raises(ValueError, match='at most 30 tables'):
                graph.visualize(sample_rows=3)
        else:
            root = ElementTree.fromstring(graph.visualize())
            assert len(root.findall(
                './{http://www.w3.org/2000/svg}g[@data-kumo-card]')) == 30
        backend.assert_not_called()
