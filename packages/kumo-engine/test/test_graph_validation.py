# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text

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


@pytest.fixture
def pdf_pages(monkeypatch):
    pages = []
    savefig = PdfPages.savefig

    def record(document, figure, *args, **kwargs):
        pages.append('\n'.join(text.get_text() for text in
                               figure.findobj(match=Text)))
        return savefig(document, figure, *args, **kwargs)

    monkeypatch.setattr(PdfPages, 'savefig', record)
    return pages


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
@pytest.mark.parametrize('suffix', ['svg', 'pdf'])
def test_visualize_exports_escaped_svg_without_source_access(
        tmp_path, alias, key, suffix, pdf_pages):
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
    destination = tmp_path / f'relationships.{suffix}'
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      side_effect=AssertionError('source accessed')), \
            patch.object(Table, 'infer_metadata',
                         side_effect=AssertionError('inference started')), \
            patch.object(Graph, 'snapshot',
                         side_effect=AssertionError('snapshot started')):
        svg = graph.visualize(destination, show_samples=False)
    assert isinstance(svg, str)
    assert svg._repr_svg_() == svg
    if suffix == 'pdf':
        data = destination.read_bytes()
        assert data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF')
        assert len(re.findall(rb'/Type /Page\b', data)) == len(pdf_pages) >= 2
        index = '\n'.join(pdf_pages[1:])
        pdf_alias = alias.replace('账户', r'\u8d26\u6237')
        pdf_key = key.replace('账户', r'\u8d26\u6237')
        assert pdf_alias in index and pdf_key in index
        assert f'purchases.{pdf_key} → {pdf_alias}.id' in index.replace('\n', '')
        assert 's3://schema-only-fixture' not in index
    else:
        assert destination.read_text(encoding='utf-8') == svg
    root = ElementTree.fromstring(svg)
    assert root.tag == '{http://www.w3.org/2000/svg}svg'
    html = ElementTree.fromstring(svg._repr_html_())
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
    assert any('Type: int64\nSemantic type: ID\nRole: FK' in tip
               for tip in details)
    assert any(f'Connected tables:\n{alias}' in tip for tip in details)
    edge = root.find('.//{http://www.w3.org/2000/svg}path[@class="kumo-edge"]')
    assert edge.get('data-dst-table') == alias
    assert edge.get('data-src-column') == key
    assert 's3://schema-only-fixture' not in svg


@pytest.mark.parametrize('count, limit', [(10, 10), (11, 5)])
@pytest.mark.parametrize('show_cols', [False, True])
def test_visualize_bounds_column_detail(count, limit, show_cols):
    users, _ = _s3_tables()
    extras = [Column(f'feature_{i:02}', Stype.numerical, Dtype.int64)
              for i in range(17)]
    table = replace(users, definition=replace(
        users.definition, time_col='event_time',
        cols=[*users.definition.cols, *extras,
              Column('event_time', Stype.timestamp, Dtype.date)]))
    graph = Graph({f'table_{i:02}': table for i in range(count)})
    root = ElementTree.fromstring(graph.visualize(show_cols=show_cols))
    visible = limit - 2 if show_cols else 0
    expected = {'id', 'event_time'} | {col.name for col in extras[:visible]}
    for card in root.findall('./{http://www.w3.org/2000/svg}g[@data-kumo-card]'):
        rows = card.findall('.//{http://www.w3.org/2000/svg}g[@data-kumo-column]')
        assert {row.get('data-kumo-column') for row in rows
                if row.get('data-kumo-column')} == expected
        time = next(row for row in rows if row.get('data-kumo-column') == 'event_time')
        assert 'Role: TS' in time.get('data-kumo-tip')
        assert re.search(rf'\+\s*{len(extras) - visible} (other )?columns',
                         ''.join(card.itertext()))


@pytest.mark.parametrize('time_column, show_cols', [
    (None, False), (None, True), ('occurred_at', False),
    ('event_id', False), ('foreign_00', True),
])
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
    card = root.find('./{http://www.w3.org/2000/svg}g[@data-kumo-card="events"]')
    rows = card.findall('.//{http://www.w3.org/2000/svg}g[@data-kumo-column]')
    expected = {'event_id'} | {column.name for column in keys}
    if time_column:
        expected.add(time_column)
    assert {row.get('data-kumo-column') for row in rows
            if row.get('data-kumo-column')} == expected
    if time_column:
        row = root.find('.//{http://www.w3.org/2000/svg}g'
                        f'[@data-kumo-column="{time_column}"]')
        role = {'event_id': 'PK/FK/TS', 'foreign_00': 'FK/TS'}.get(time_column, 'TS')
        assert f'Role: {role}' in row.get('data-kumo-tip')


@pytest.mark.parametrize('layout', ['standard', 'compact'])
def test_visualize_layout_is_deterministic_with_cycles_and_isolates(layout):
    _, table = _s3_tables()
    tables = {
        name: replace(table, definition=replace(
            table.definition, pkey=key,
            cols=[*table.definition.cols, Column(key, Stype.ID, Dtype.int64)]))
        for name, key in [('a', 'b.c'), ('a.b', 'c')]
    } | {'c': table, 'isolated': table}
    edges = [
        Edge('c', 'user_id', 'a.b'), Edge('c', 'event_id', 'a'),
        Edge('a.b', 'user_id', 'c'), Edge('a', 'event_id', 'c'),
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


@pytest.mark.parametrize('layout', ['auto', 'standard', 'compact'])
def test_visualize_empty_graph(layout):
    svg = Graph([]).visualize(layout=layout)
    root = ElementTree.fromstring(svg)
    assert 'No tables' in ''.join(root.itertext())
    assert all(float(value) > 0 for value in root.get('viewBox').split()[2:])
    payload = ElementTree.fromstring(svg._repr_html_()).find(
        './script[@type="application/json"]')
    assert json.loads(payload.text)['initial'] == (
        'standard' if layout == 'auto' else layout)


def test_widget_shell_composes_non_graph_content_and_escapes_data():
    from kumo.engine._widget import button, render_widget

    label = '</script><script>alert("chart")</script> & values'
    body = ('<div class="kumo-widget-toolbar">' +
            button('chart-toggle', label, text='<Toggle>') + '</div>'
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 30">'
            '<rect width="75" height="20"/></svg>')
    html, repeated = [
        render_widget(body, data={'labels': [label], 'values': [0.75]},
                      attributes={'aria-label': label}) for _ in range(2)
    ]
    root = ElementTree.fromstring(html)
    assert root.get('id') != ElementTree.fromstring(repeated).get('id')
    with pytest.raises(ValueError, match='data-.*aria-'):
        render_widget(body, attributes={'onclick': 'alert(1)'})
    assert root.get('aria-label') == label
    control = root.find('.//button')
    assert control.get('aria-label') == label
    assert control.text == '<Toggle>'
    assert 'disabled' in control.attrib
    payload = root.find('.//script[@type="application/json"]')
    assert json.loads(payload.text) == {'labels': [label], 'values': [0.75]}
    assert root.find('.//{http://www.w3.org/2000/svg}rect').get('width') == '75'
    assert '<script>alert(' not in html


@pytest.mark.parametrize('layout',
                         [None, [], 'original', 'circular'])
def test_visualize_rejects_invalid_layout_before_source_access(layout):
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      side_effect=AssertionError('source accessed')):
        with pytest.raises(ValueError, match='layout must be'):
            Graph(_s3_tables()).visualize(layout=layout, show_samples=True)


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
        payload = json.loads(ElementTree.fromstring(svg._repr_html_()).find(
            './script[@type="application/json"]').text)
        assert payload['initial'] == layout
        selected = payload['layouts'][layout]
        pdf = svg._pdf(layout)
        assert pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF')
        assert len(re.findall(rb'/Type /Page\b', pdf)) >= 2
        assert list(map(float, root.get('viewBox').split())) == [
            0, 0, selected['width'], selected['height']]
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


def test_visualize_pdf_indexes_hidden_columns(tmp_path, pdf_pages):
    users, events = _s3_tables()
    users = replace(users, definition=replace(
        users.definition, time_col='hidden_date', cols=[
            *users.definition.cols,
            Column('hidden_date', Stype.timestamp, Dtype.date),
            *(Column(f'hidden_{i}_é_Ω_账户', Stype.numerical, Dtype.float64)
              for i in range(100)),
        ]))
    graph = Graph([users, events], [Edge('events', 'user_id', 'users')])
    destination = tmp_path / 'graph.PDF'
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      side_effect=AssertionError('source accessed')):
        svg = graph.visualize(destination, show_cols=False)
    data = destination.read_bytes()
    assert len(re.findall(rb'/Type /Page\b', data)) == len(pdf_pages) > 2
    index = '\n'.join(pdf_pages[1:])
    assert 'hidden_99_é_Ω_账户' not in svg
    assert all(f'hidden_{i}_é_Ω_\\u8d26\\u6237' in index for i in range(100))
    assert 'hidden_date' in index and 'TS' in index
    assert 'float64' in index and 'events.user_id → users.id' in index


@pytest.mark.parametrize('suffix', ['md', 'MARKDOWN'])
def test_visualize_markdown_exports_complete_safe_schema(tmp_path, suffix):
    users, events = _s3_tables()
    alias = 'end"]\n```<script>alert(1)</script>|账户 www.example.com a@b.com'
    key = 'user_id|[link](https://example.com)'
    events = replace(events, definition=replace(events.definition, cols=[
        events.definition.cols[0], Column(key, Stype.ID, Dtype.int64),
        Column('hidden_**value**', Stype.numerical, Dtype.float64)]))
    graph = Graph({alias: users, 'events': events},
                  [Edge('events', key, alias)])
    destination = tmp_path / f'graph.{suffix}'
    destination.write_text('stale contents')
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      side_effect=AssertionError('source accessed')):
        svg = graph.visualize(destination, show_cols=False)
    markdown = destination.read_text(encoding='utf-8')
    assert markdown.startswith('# Table relationships\n')
    assert markdown.count('```') == 2
    assert 'flowchart LR\n' in markdown
    assert 'table_1 -->|"' in markdown and '| table_0' in markdown
    assert 'end#34;]<br/>#96;#96;#96;#60;script#62;' in markdown
    schema = markdown.split('```')[-1]
    assert '<script>' not in markdown and '[link](' not in schema
    assert 'https://example.com' not in schema
    assert r'www\.example\.com a\@b\.com' in schema
    assert 'hidden_**value**' not in svg
    assert r'| hidden\_\*\*value\*\* | float64 | Column |' in markdown
    assert '账户' in markdown and 'FK' in markdown and 'PK' in markdown
    assert 'stale contents' not in markdown
    assert svg._markdown() == markdown
    assert Graph([]).visualize(tmp_path / 'empty.md')._markdown().endswith(
        'No tables.\n')


@pytest.mark.parametrize('show_samples', [0, 1, 6, 'true', None])
def test_visualize_rejects_invalid_show_samples(show_samples):
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      side_effect=AssertionError('source accessed')):
        with pytest.raises(ValueError, match='show_samples'):
            Graph(_s3_tables()).visualize(show_samples=show_samples)


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
    definition = tables[0].definition
    assert definition is not None
    tables[0] = replace(tables[0], definition=replace(
        definition, cols=[
            *definition.cols,
            Column('created_at', Stype.timestamp, Dtype.date)]))
    frame = pd.DataFrame({
        'id': [secret, None, ['alpha', 'beta'], 'later non-null',
               'fourth', 'fifth', 'sixth', 'outside cap'],
        'event_id': ['x' * 1000, None, 'third', 'later non-null',
                     'fourth', 'fifth', 'sixth', 'outside cap'],
        'user_id': list(range(1, 9)),
        'created_at': pd.date_range('2026-10-01 09:00:00.123', periods=8,
                                    tz='UTC'),
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
        svg = Graph(tables).visualize(destination, show_samples=True,
                                      show_cols=True)
        html = svg._repr_html_()
        if outcome == 'values':
            repeat = svg._repr_html_()
            assert re.sub(r'kumo-[0-9a-f]{32}', 'kumo-ID',
                          html) == re.sub(r'kumo-[0-9a-f]{32}', 'kumo-ID', repeat)
            assert set(re.findall(r' id="([^"]+)"', html)).isdisjoint(
                re.findall(r' id="([^"]+)"', repeat))
    assert backend.get_sample.call_count == 2
    for call, table in zip(backend.get_sample.call_args_list,
                           sorted(tables, key=lambda table: table.table_name)):
        assert call.args == (table.connector, table.table_name, 30)
        assert call.kwargs['sql'] == (sql if table.table_name == 'users' else None)
        assert call.kwargs['timeout'] > 0
    assert 'Sample values' not in svg
    markdown = svg._markdown()
    assert 'private sample' not in markdown and 'credential secret' not in markdown
    assert svg._repr_svg_() == destination.read_text() == str(svg)
    assert 'credential secret' not in html
    assert '<script>alert' not in html
    tips = ElementTree.fromstring(html).findall(
        './/{http://www.w3.org/2000/svg}g[@data-kumo-tip]')
    details = '\n'.join(tip.get('data-kumo-tip') for tip in tips)
    if outcome in ('values', 'sql'):
        assert '2026-10-01 09:00:00.123000+00:00' in details
        assert 'Timestamp(' not in details
        assert secret in details and '\nnull' not in details
        assert "['alpha', 'beta']" in details
        assert 'later non-null' in details and 'sixth' in details
        assert 'outside cap' not in details
        for tip in tips:
            if 'Sample values:\n' in tip.get('data-kumo-tip'):
                assert len(tip.get('data-kumo-tip').split(
                    'Sample values:\n')[1].splitlines()) == 6
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


def test_visualize_samples_bound_preview_and_select_columns_independently():
    import pandas as pd

    from kumo.engine._visualization import _preview

    table = _s3_tables()[0]
    backend = Mock()
    backend.get_sample.return_value.sample_rows = pd.DataFrame({
        'id': [None] * 29 + ['last allowed', 'outside preview'],
        'other': ['first', None, 'second', 'third', 'fourth', 'fifth',
                  'sixth', 'outside cap'] + [None] * 23,
    }).to_json(orient='table')
    with patch.object(S3Connector, 'source_backend', new_callable=PropertyMock,
                      return_value=backend):
        preview = _preview(table, ['id', 'other'])
    assert preview['id'] == "\nSample values:\n'last allowed'"
    assert preview['other'].splitlines()[2:] == [
        "'first'", "'second'", "'third'", "'fourth'", "'fifth'", "'sixth'"
    ]


@pytest.mark.parametrize('count', [30, 31])
def test_visualize_table_cap_precedes_source_access(count):
    table = _s3_tables()[0]
    graph = Graph({f'table_{i:02}': table for i in range(count)})
    with patch.object(S3Connector, 'source_backend',
                      new_callable=PropertyMock) as backend:
        if count == 31:
            with pytest.raises(ValueError, match='at most 30 tables'):
                graph.visualize(show_samples=True)
        else:
            root = ElementTree.fromstring(graph.visualize())
            assert len(root.findall(
                './{http://www.w3.org/2000/svg}g[@data-kumo-card]')) == 30
        backend.assert_not_called()
