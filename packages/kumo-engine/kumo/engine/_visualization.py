# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SVG rendering of declared table relationships."""

import json
import re
from datetime import date
from html import escape
from reprlib import Repr
from typing import Mapping, Sequence
from unicodedata import combining
from uuid import uuid4
from xml.etree import ElementTree

from kumoapi.graph import Edge, validate_edge_endpoints
from kumoapi.source_table import CustomSQLSourceTable
from kumoapi.table import TableDefinition

from kumo.engine import _widget
from kumo.engine._widget import THEME, render_anywidget, render_widget
from kumo.engine.table import Table

_COLUMN_LIMIT = 10
_LARGE_GRAPH_COLUMN_LIMIT = 5


class SVG(str):
    """SVG text with optional notebook-only layout controls and previews."""

    def __new__(cls, value: str, samples: dict[str, str] | None = None,
                namespace: str = '', layouts: dict | None = None,
                initial_layout: str = 'standard', index: list | None = None):
        obj = super().__new__(cls, value)
        obj._samples = samples or {}
        obj._namespace = namespace
        obj._layouts = layouts or {}
        obj._initial_layout = initial_layout
        obj._index = index or []
        obj._pdfs = {}
        obj._widget = None
        return obj

    def _repr_svg_(self) -> str:
        return str(self)

    def _pdf(self, layout: str) -> bytes:
        from kumo.engine._visualization_pdf import render_pdf

        if layout not in self._pdfs:
            self._pdfs[layout] = render_pdf(self, self._layouts[layout],
                                            self._index)
        return self._pdfs[layout]

    def _markdown(self) -> str:
        def label(value, mermaid=False):
            value = value.replace('\r\n', '\n').replace('\r', '\n')
            if mermaid:
                return ''.join(
                    '<br/>' if char == '\n' else
                    f'#{ord(char)};' if char in '#&"<>\\`|' else char
                    for char in value)
            return re.sub(r'([\\`*_\[\]|#.@:])', r'\\\1',
                          escape(value, quote=False)).replace('\n', '<br/>')

        lines = ['# Table relationships', '']
        if not self._index:
            return '\n'.join(lines + ['No tables.', ''])
        ids = {table['name']: f'table_{i}'
               for i, table in enumerate(self._index)}
        lines += ['Arrows point from foreign keys to referenced primary keys.',
                  'Sample values are excluded.', '', '```mermaid',
                  'flowchart LR']
        lines.extend(f'    {ids[name]}["{label(name, True)}"]' for name in ids)
        root = ElementTree.fromstring(self)
        for edge in root.findall(
                './/{http://www.w3.org/2000/svg}path[@class="kumo-edge"]'):
            source = ids[edge.get('data-src-table')]
            target = ids[edge.get('data-dst-table')]
            columns = (f'{edge.get("data-src-column")} → '
                       f'{edge.get("data-dst-column")}')
            lines.append(f'    {source} -->|"{label(columns, True)}"| {target}')
        lines += ['```', '', '## Schema', '',
                  'All declared columns are listed. PK: primary key; '
                  'FK: foreign key; TS: time column; TE: end-time column.', '']
        for table in self._index:
            lines += [f'### {label(table["name"])}', '',
                      '| Column | Data type | Role |', '| --- | --- | --- |']
            for column in table['columns']:
                cells = [column['name'], column['type'],
                         column['role'] or 'Column']
                lines.append('| ' + ' | '.join(map(label, cells)) + ' |')
            if not table['columns']:
                lines += ['', 'No columns.']
            lines += ['', '**Outgoing connections**', '']
            lines.extend(f'- {label(connection)}' for connection in
                         table['connections'] or ['No outgoing connections'])
            lines.append('')
        return '\n'.join(lines)

    def as_widget(self):
        """Return a cached widget model with independent frontend views."""
        if self._widget is None or self._widget.comm is None:
            self._widget = render_anywidget(**self._viewer())
        return self._widget

    def _ipython_display_(self):
        if not _widget.is_snowflake_workspace():
            raise NotImplementedError
        from IPython.display import display

        try:
            widget = self.as_widget()
        except ImportError as error:
            display({'text/html': _widget.render_static_svg(
                self, str(error), 'Table relationships'),
                'text/plain': str(error)}, raw=True)
        else:
            display(widget)

    def _repr_html_(self) -> str:
        return render_widget(**self._viewer())

    def _viewer(self) -> dict:
        from kumo.engine._visualization_viewer import viewer

        svg = str(self)
        for metadata, sampled in self._samples.items():
            svg = svg.replace(metadata, sampled)
        return dict(viewer(svg, self._layouts, self._initial_layout, str(self)),
                    namespace=self._namespace)


def _wrap(text: str, width: int) -> list[str]:
    lines, line, size = [], '', 0
    split, split_size = 0, 0
    for char in text:
        code = ord(char)
        if not (code in (9, 10, 13) or 32 <= code <= 0xD7FF
                or 0xE000 <= code <= 0xFFFD or 0x10000 <= code <= 0x10FFFF):
            raise ValueError('Table and column names must contain valid XML '
                             'characters to visualize them.')
        char_size = 0 if combining(char) else (
            2 if code > 127 or char.isupper() or char in 'mw@%&#' else 1)
        while line and size + char_size > width:
            if split:
                lines.append(line[:split])
                line, size = line[split:], size - split_size
            else:
                lines.append(line)
                line, size = '', 0
            split, split_size = 0, 0
        line += char
        size += char_size
        if char.isspace() or char in '_-':
            split, split_size = len(line), size
    return lines + [line]


def _squircle(width: float, height: float, radius: float = 18,
              *, square_bottom: bool = False) -> str:
    # Cubic corners flatten at the sides, unlike circular rounded rectangles.
    r = min(radius, width / 2, height / 2)
    bottom = (f'V {height} H 0 ' if square_bottom else
              f'V {height - r} C {width} {height} {width} {height} '
              f'{width - r} {height} H {r} C 0 {height} 0 {height} '
              f'0 {height - r} ')
    return (f'M {r} 0 H {width - r} C {width} 0 {width} 0 {width} {r} '
            f'{bottom}V {r} C 0 0 0 0 {r} 0 Z')


def _attribute(value: str) -> str:
    return (escape(value).replace('\t', '&#9;').replace('\n', '&#10;')
            .replace('\r', '&#13;'))


def _tooltip(text: str, row: str = '', table: str = '',
             column: str = '',
             connected_tables: Sequence[str] = ()) -> str:
    connections = escape(json.dumps(list(connected_tables)))
    attr = _attribute(text)
    text = escape(text)
    table = _attribute(table)
    column = _attribute(column)
    return (f'<g tabindex="0" role="group" data-kumo-row="{row}" '
            f'data-kumo-table="{table}" data-kumo-column="{column}" '
            f'data-kumo-connections="{connections}" '
            f'aria-label="{attr}" '
            f'data-kumo-tip="{attr}"><title>{text}</title>')


def _preview(table: Table, columns: Sequence[str]) -> dict[str, str]:
    import pandas as pd
    from kumo_core.sources.preview import sample_dataframe

    fallback = dict.fromkeys(columns, '\nSample values: unavailable')
    try:
        assert table.definition is not None
        source = table.definition.source_table
        sql = source.sql if isinstance(source, CustomSQLSourceTable) else None
        response = table.connector.source_backend.get_sample(
            table.connector, table.table_name, 30, sql=sql, timeout=10)
        frame = sample_dataframe(response.sample_rows)
        if frame is None:
            return fallback
        short = Repr()
        short.maxstring, short.maxlist, short.maxtuple = 120, 5, 5
        for column in columns:
            if column not in frame:
                continue
            values = frame[column].iloc[:30].tolist()
            if not values:
                fallback[column] = '\nSample values: no rows returned'
                continue
            labels = []
            for value in values:
                null = (value is None or value is pd.NA or value is pd.NaT
                        or isinstance(value, float) and value != value)
                if null:
                    continue
                label = (str(value) if isinstance(value, date) else
                         short.repr(value))
                labels.append(label if len(label) <= 160 else label[:157] +
                              '...')
                if len(labels) == 6:
                    break
            fallback[column] = (
                '\nSample values:\n' + '\n'.join(labels) if labels else
                '\nSample values: No non-null values in preview')
    except Exception:
        # Preview failures must not hide the schema or expose source secrets.
        pass
    return fallback


def render(tables: Mapping[str, Table], edges: Sequence[Edge], *,
           show_cols: bool, show_samples: bool = False, layout: str = 'auto') -> SVG:
    from kumo.engine._visualization_layout import build_layouts, choose_layout

    if not isinstance(layout, str) or layout not in (
            'auto', 'standard', 'compact'):
        raise ValueError('layout must be auto, standard, or compact.')
    if type(show_samples) is not bool:
        raise ValueError('show_samples must be True or False.')
    if len(tables) > 30:
        raise ValueError('Graph visualization supports at most 30 tables; '
                         'select a smaller graph to visualize.')
    validate_edge_endpoints(tables, edges)
    names = sorted(tables)
    edges = sorted(edges, key=lambda e: (e.src_table, e.fkey, e.dst_table))
    definitions: dict[str, TableDefinition] = {}
    keys = {name: {} for name in names}
    neighbors: dict[str, set[str]] = {name: set() for name in names}
    for name in names:
        definition = tables[name].definition
        if definition is None:
            raise ValueError(
                f'Table {name!r} has no definition; infer metadata '
                'or supply its definition before visualizing.')
        definitions[name] = definition
        columns = [col.name for col in definition.cols]
        if len(columns) != len(set(columns)):
            raise ValueError(f'Table {name!r} has duplicate column names; '
                             'give each column a unique name.')
        if definition.pkey is not None:
            if definition.pkey not in columns:
                raise ValueError(f'Primary key {name}.{definition.pkey} is '
                                 'missing from the table columns.')
            keys[name][definition.pkey] = 'PK'
    for edge in edges:
        src, dst = edge.src_table, edge.dst_table
        if src == dst:
            raise ValueError(f'Self-relation {src}.{edge.fkey} -> {dst} cannot '
                             'be visualized; self-relations are not supported '
                             'by graph snapshots. Remove this edge first.')
        if edge.fkey not in {col.name for col in definitions[src].cols}:
            raise ValueError(f'Foreign key {src}.{edge.fkey} is missing '
                             'from the table columns.')
        if definitions[dst].pkey is None:
            raise ValueError(f'Destination table {dst!r} has no primary key; '
                             'declare one before visualizing this edge.')
        keys[src][edge.fkey] = ('PK/FK'
                                if definitions[src].pkey == edge.fkey else 'FK')
        neighbors[src].add(dst)
        neighbors[dst].add(src)

    for name, definition in definitions.items():
        for column, marker, label in (
                (definition.time_col, 'TS', 'Time'),
                (definition.end_time_col, 'TE', 'End-time')):
            if column is not None:
                if column not in {col.name for col in definition.cols}:
                    raise ValueError(f'{label} column {name}.{column} is '
                                     'missing from the table columns.')
                role = keys[name].get(column, '')
                keys[name][column] = f'{role}/{marker}' if role else marker

    cards = {}
    for name in names:
        definition = definitions[name]
        pkey = definition.pkey
        columns = sorted((col.name for col in definition.cols), key=lambda col:
                         (col != pkey, 'FK' not in keys[name].get(col, ''),
                          col not in keys[name], col))
        limit = (_COLUMN_LIMIT
                 if len(names) <= 10 else _LARGE_GRAPH_COLUMN_LIMIT)
        columns = columns[:max(len(keys[name]), limit if show_cols else 0)]
        title = _wrap(name, 27)
        header = 20 + 22 * len(title)
        rows = []
        ports = {}
        y = header + 8
        for column in columns:
            # Reserve extra room when a field carries multiple role markers.
            lines = _wrap(column, 23 if len(keys[name].get(column, '')) > 5
                          else 25)
            height = 12 + 20 * len(lines)
            ports[column] = y + height / 2
            rows.append((column, lines, y, height))
            y += height
        hidden = len(definition.cols) - len(columns)
        footer = (f'+ {hidden} other column' +
                  ('s' if hidden != 1 else '') if hidden else
                  ('No columns' if not columns else ''))
        height = y + (26 if footer else 4)
        cards[name] = (title, header, rows, ports, footer, height)

    # Rank primary-key tables before their dependents.
    parents = {name: set() for name in names}
    for edge in edges:
        parents[edge.src_table].add(edge.dst_table)
    ranks: dict[str, int] = {}
    pending = set(names)
    while pending:
        ready = [name for name in pending if not parents[name] & pending]
        name = min(ready or pending, key=lambda n: (-len(neighbors[n]), n))
        ranks[name] = max(
            (ranks[parent] + 1 for parent in parents[name] if parent in ranks),
            default=0)
        pending.remove(name)
    count = max(ranks.values(), default=0) + 1
    layers = [[] for _ in range(count)]
    heights = [cards[name][-1] for name in names]
    for node, name in enumerate(names):
        layers[ranks[name]].append(node)
    node_ids = {name: node for node, name in enumerate(names)}
    routes = []
    adjacent: dict[int, list[int]] = {node: [] for node in range(len(names))}
    for edge in edges:
        source, target = ranks[edge.dst_table], ranks[edge.src_table]
        route = [node_ids[edge.dst_table]]
        # Empty slots carry long edges through a rank without crossing a card.
        for rank in range(source + 1, target):
            node = len(heights)
            heights.append(16)
            layers[rank].append(node)
            adjacent[node] = []
            route.append(node)
        route.append(node_ids[edge.src_table])
        routes.append(route)
        if source < target:
            for a, b in zip(route, route[1:]):
                adjacent[a].append(b)
                adjacent[b].append(a)
    centers = {
        node: index
        for layer in layers
        for index, node in enumerate(layer)
    }
    for sweep in (list(range(count)), list(reversed(range(count)))) * 2:
        for rank in sweep:
            layers[rank].sort(
                key=lambda node: (sum(centers[n] for n in adjacent[node]) / len(
                    adjacent[node]) if adjacent[node] else centers[node], node))
            offset = 0
            for node in layers[rank]:
                centers[node] = offset + heights[node] / 2
                offset += heights[node] + 50
    layer_heights = [
        sum(heights[node] for node in layer) + max(0,
                                                   len(layer) - 1) * 50
        for layer in layers
    ]
    feedback = sum(ranks[e.dst_table] >= ranks[e.src_table] for e in edges)
    card_width, gap = 304, max(120, 60 + 8 * feedback)
    margin = max(80, 32 + 6 * feedback)
    top = 40 + 24 * feedback
    canvas_height = max(layer_heights, default=90)
    xy = {}
    for rank, layer in enumerate(layers):
        y = top + (canvas_height - layer_heights[rank]) / 2
        for node in layer:
            xy[node] = (margin + rank * (card_width + gap), y)
            y += heights[node] + 50
    width = count * card_width + (count - 1) * gap + 2 * margin
    height = top + canvas_height + 40
    if not names:
        width, height = 304, 112
    standard_paths = []
    lane = 0
    handles = set()
    for index, edge in enumerate(edges):
        route = routes[index]
        sx, sy = xy[route[0]]
        dx, dy = xy[route[-1]]
        sx += card_width
        sy += cards[edge.dst_table][3][definitions[edge.dst_table].pkey]
        dy += cards[edge.src_table][3][edge.fkey]
        path = f'M {sx} {sy}'
        if ranks[edge.dst_table] < ranks[edge.src_table]:
            x, y = sx, sy
            for node in route[1:]:
                nx, ny = (dx, dy) if node == route[-1] else (xy[node][0],
                                                             xy[node][1] +
                                                             heights[node] / 2)
                # Use horizontal Bézier controls at half the column gap.
                half = (nx - x) / 2
                path += f' C {x + half} {y} {nx - half} {ny} {nx} {ny}'
                x, y = nx, ny
                if node != route[-1]:
                    path += (f' C {x + card_width / 3} {y} '
                             f'{x + 2 * card_width / 3} {y} '
                             f'{x + card_width} {y}')
                    x += card_width
        else:
            overhead = 20 + lane * 24
            bow = 42 + lane * 8
            lane += 1
            path += (f' C {sx + bow} {sy} {sx + bow} {overhead} '
                     f'{sx} {overhead} C {sx - gap} {overhead} '
                     f'{dx + gap} {overhead} {dx} {overhead} '
                     f'C {dx - bow} {overhead} {dx - bow} {dy} {dx} {dy}')
        standard_paths.append(path)
        handles.update(((sx, sy), (dx, dy)))

    layout_cards = [dict(name=name, width=card_width,
                         height=cards[name][-1], rank=ranks[name],
                         x=xy[node_ids[name]][0], y=xy[node_ids[name]][1])
                    for name in names]
    layout_edges = [dict(source=edge.dst_table, target=edge.src_table,
                         source_port=cards[edge.dst_table][3][
                             definitions[edge.dst_table].pkey],
                         target_port=cards[edge.src_table][3][edge.fkey])
                    for edge in edges]
    standard = dict(width=width, height=height,
                    positions={name: list(xy[node_ids[name]]) for name in names},
                    paths=standard_paths, handles=sorted(handles))
    layouts = build_layouts(layout_cards, layout_edges, standard)
    selected = (choose_layout(layout_cards, layout_edges, layouts)
                if layout == 'auto' else layout)
    geometry = layouts[selected]
    width, height = geometry['width'], geometry['height']
    samples = {}
    uid = 'kumo-' + uuid4().hex
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'role="group" aria-label="Table relationships" '
        f'font-family="{THEME["font"]}">',
        '<title>Table relationships</title>',
        '<desc>Lines connect foreign keys (FK) to primary keys (PK).'
        ' TS marks the declared time column; TE marks the end-time column.'
        '</desc>', f'<defs><pattern id="{uid}-grid" width="16" height="16" '
        'patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.7" '
        f'fill="{THEME["grid"]}"/></pattern></defs>',
        f'<rect width="{width}" height="{height}" fill="{THEME["background"]}"/>',
        f'<rect width="{width}" height="{height}" fill="url(#{uid}-grid)"/>'
    ]
    if not names:
        svg.append('<text class="kumo-muted" x="24" y="60" font-size="16" fill="#606060">'
                   'No tables</text>')
    for index, edge in enumerate(edges):
        path = geometry['paths'][index]
        label = _attribute(
            f'{edge.src_table}.{edge.fkey} → '
            f'{edge.dst_table}.{definitions[edge.dst_table].pkey}')
        endpoints = ' '.join(
            f'data-{key}="{_attribute(value)}"'
            for key, value in [('src-table',
                                edge.src_table), (
                                    'src-column',
                                    edge.fkey), ('dst-table', edge.dst_table),
                               ('dst-column',
                                definitions[edge.dst_table].pkey)])
        svg.append(
            f'<g role="img" aria-label="{label}"><title>{label}</title>'
            f'<path class="kumo-edge" {endpoints} d="{path}" fill="none" '
            f'stroke="{THEME["edge"]}" stroke-width="1"/></g>')
    for name in names:
        x, y = geometry['positions'][name]
        title, header, rows, ports, footer, card_height = cards[name]
        definition = definitions[name]
        preview = (_preview(tables[name], [row[0] for row in rows])
                   if show_samples and rows else {})
        outgoing = [e for e in edges if e.src_table == name]
        connected = sorted(neighbors[name])
        summary = (f'{name}\n{len(definition.cols)} '
                   f'column{"s" if len(definition.cols) != 1 else ""}')
        if connected:
            summary += '\nConnected tables:\n' + '\n'.join(connected)
        else:
            summary += '\nNo connected tables'
        shape = _squircle(card_width, card_height)
        clip = f'{uid}-card-{node_ids[name]}'
        svg.append(f'<g data-kumo-card="{_attribute(name)}" '
                   f'transform="translate({x} {y})">'
                   f'<defs><clipPath id="{clip}"><path d="{shape}"/>'
                   '</clipPath></defs>'
                   f'<path class="kumo-card" d="{shape}" fill="{THEME["card_background"]}" stroke="{THEME["card_border"]}"/>')
        svg.append(_tooltip(summary, table=name, connected_tables=connected))
        svg.append(f'<path class="kumo-surface" '
                   f'd="{_squircle(card_width, header, square_bottom=True)}" '
                   f'clip-path="url(#{clip})" fill="{THEME["header_background"]}"/>')
        for i, line in enumerate(title):
            svg.append('<text class="kumo-card-name" xml:space="preserve" '
                       f'x="16" y="{26 + 22 * i}" font-size="17" '
                       f'font-weight="700" fill="{THEME["heading"]}">{escape(line)}'
                       '</text>')
        svg.append('</g>')
        for row_index, (column, lines, ry, rh) in enumerate(rows):
            role = keys[name].get(column, '')
            col = next(c for c in definition.cols if c.name == column)
            detail = (f'{name}.{column}\nType: {col.dtype.value}\n'
                      f'Semantic type: {col.stype.value}\n'
                      f'Role: {role or "Column"}')
            detail += ''.join(f'\nReferences: {e.dst_table}.'
                              f'{definitions[e.dst_table].pkey}'
                              for e in outgoing if e.fkey == column)
            row = f'{node_ids[name]}-{row_index}'
            metadata = _tooltip(detail, row, name, column)
            if column in preview:
                samples[metadata] = _tooltip(detail + preview[column], row,
                                             name, column)
            svg.append(metadata)
            fill = THEME['role_background'] if role else 'transparent'
            cls = 'kumo-surface kumo-key-surface' if role else 'kumo-surface'
            svg.append(f'<path class="{cls}" '
                       f'transform="translate(10 {ry + 2})" '
                       f'd="{_squircle(card_width - 20, rh - 4, 7)}" '
                       f'fill="{fill}"/>')
            for i, line in enumerate(lines):
                svg.append('<text class="kumo-column-name" xml:space="preserve" '
                           f'x="18" y="{ry + 21 + 20 * i}" '
                           f'font-size="16" fill="{THEME["foreground"]}">{escape(line)}'
                           '</text>')
            if role:
                svg.append(f'<text class="kumo-role" x="{card_width - 18}" '
                           f'y="{ry + rh / 2 + 4}" text-anchor="end" '
                           f'font-size="10" font-weight="700" '
                           f'fill="{THEME["accent"]}">{role}</text>')
            svg.append('</g>')
        if footer:
            svg.append(f'<text class="kumo-muted" x="18" y="{card_height - 14}" font-size="12" '
                       f'fill="{THEME["muted"]}">{footer}</text>')
        svg.append('</g>')
    for x, y in geometry['handles']:
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{THEME["muted"]}"/>')
    index = [dict(
        name=name,
        columns=[dict(name=col.name, type=col.dtype.value,
                      role=keys[name].get(col.name, ''))
                 for col in definitions[name].cols],
        connections=[f'{edge.src_table}.{edge.fkey} → '
                     f'{edge.dst_table}.{definitions[edge.dst_table].pkey}'
                     for edge in edges if edge.src_table == name],
    ) for name in names]
    return SVG(''.join(svg) + '</svg>', samples, uid, layouts=layouts,
               initial_layout=selected, index=index)
