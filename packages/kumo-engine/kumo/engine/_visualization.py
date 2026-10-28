# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free SVG rendering of graph schema metadata."""

import json
from html import escape
from reprlib import Repr
from typing import Mapping, Sequence
from unicodedata import combining
from uuid import uuid4

from kumoapi.graph import Edge, validate_edge_endpoints
from kumoapi.source_table import CustomSQLSourceTable
from kumoapi.table import TableDefinition

from kumo.engine._widget import THEME
from kumo.engine.table import Table

_COLUMN_LIMIT = 10
_LARGE_GRAPH_COLUMN_LIMIT = 5


class SVG(str):
    """SVG text with optional notebook-only layout controls and previews."""

    def __new__(cls, value: str, samples: dict[str, str] | None = None,
                namespace: str = '', layouts: dict | None = None,
                initial_layout: str = 'standard'):
        obj = super().__new__(cls, value)
        obj._samples = samples or {}
        obj._namespace = namespace
        obj._layouts = layouts or {}
        obj._initial_layout = initial_layout
        return obj

    def _repr_svg_(self) -> str:
        return str(self)

    def _repr_html_(self, *, _highlight_table: bool = False) -> str:
        from kumo.engine._visualization_viewer import render_html

        svg = str(self)
        for metadata, sampled in self._samples.items():
            svg = svg.replace(metadata, sampled)
        if self._namespace:
            svg = svg.replace(self._namespace, 'kumo-' + uuid4().hex)
        return render_html(svg, 'table' if _highlight_table else 'row',
                           self._layouts, self._initial_layout)


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


def _tooltip(text: str, row: str = '', table: str = '',
             column: str = '',
             connected_tables: Sequence[str] = ()) -> str:
    connections = escape(json.dumps(list(connected_tables)))
    text = escape(text)
    attr = text.replace('\n', '&#10;')
    table = escape(table).replace('\n', '&#10;')
    column = escape(column).replace('\n', '&#10;')
    return (f'<g tabindex="0" role="group" data-kumo-row="{row}" '
            f'data-kumo-table="{table}" data-kumo-column="{column}" '
            f'data-kumo-connections="{connections}" '
            f'aria-label="{attr}" '
            f'data-kumo-tip="{attr}"><title>{text}</title>')


def _preview(table: Table, count: int,
             columns: Sequence[str]) -> dict[str, str]:
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
                label = short.repr(value)
                labels.append(label if len(label) <= 160 else label[:157] +
                              '...')
                if len(labels) == count:
                    break
            fallback[column] = (
                '\nSample values:\n' + '\n'.join(labels) if labels else
                '\nSample values: No non-null values in preview')
    except Exception:
        # Preview failures must not hide the schema or expose source secrets.
        pass
    return fallback


def render(tables: Mapping[str, Table], edges: Sequence[Edge], *,
           show_cols: bool, sample_rows: int = 0, layout: str = 'auto') -> SVG:
    from kumo.engine._visualization_layout import build_layouts, choose_layout

    if not isinstance(layout, str) or layout not in (
            'auto', 'standard', 'compact'):
        raise ValueError('layout must be auto, standard, or compact.')
    if type(sample_rows) is not int or not 0 <= sample_rows <= 3:
        raise ValueError('sample_rows must be an integer from 0 to 3.')
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
        if definition.time_col is not None:
            if definition.time_col not in {col.name for col in definition.cols}:
                raise ValueError(f'Time column {name}.{definition.time_col} is '
                                 'missing from the table columns.')
            role = keys[name].get(definition.time_col, '')
            keys[name][definition.time_col] = f'{role}/T' if role else 'T'

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
            # Reserve extra room when a field carries all three role markers.
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

    # Rank primary-key tables before their dependents, as in the UI's LR flow.
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
                # UI getBezierPath: horizontal handles, half-gap controls.
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
        ' T marks the declared time column.'
        '</desc>', f'<defs><pattern id="{uid}-grid" width="16" height="16" '
        'patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.7" '
        f'fill="{THEME["grid"]}"/></pattern></defs>',
        f'<rect width="{width}" height="{height}" fill="{THEME["background"]}"/>',
        f'<rect width="{width}" height="{height}" fill="url(#{uid}-grid)"/>'
    ]
    if not names:
        svg.append('<text x="24" y="60" font-size="16" fill="#606060">'
                   'No tables</text>')
    for index, edge in enumerate(edges):
        path = geometry['paths'][index]
        label = escape(f'{edge.src_table}.{edge.fkey} → '
                       f'{edge.dst_table}.{definitions[edge.dst_table].pkey}')
        endpoints = (f'data-src-port="{cards[edge.src_table][3][edge.fkey]}" '
                     f'data-dst-port="{cards[edge.dst_table][3][definitions[edge.dst_table].pkey]}" ') + ' '.join(
            f'data-{key}="{escape(value).replace(chr(10), "&#10;")}"'
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
        preview = (_preview(tables[name], sample_rows, [row[0] for row in rows])
                   if sample_rows and rows else {})
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
        svg.append(f'<g data-kumo-card="{escape(name).replace(chr(10), "&#10;")}" '
                   f'data-width="{card_width}" data-height="{card_height}" '
                   f'data-rank="{ranks[name]}" '
                   f'transform="translate({x} {y})">'
                   f'<defs><clipPath id="{clip}"><path d="{shape}"/>'
                   '</clipPath></defs>'
                   f'<path d="{shape}" fill="{THEME["background"]}" stroke="{THEME["card_border"]}"/>')
        svg.append(_tooltip(summary, table=name, connected_tables=connected))
        svg.append(f'<path class="kumo-surface" '
                   f'd="{_squircle(card_width, header, square_bottom=True)}" '
                   f'clip-path="url(#{clip})" fill="{THEME["header_background"]}" '
                   f'stroke="{THEME["header_border"]}"/>')
        for i, line in enumerate(title):
            svg.append('<text xml:space="preserve" '
                       f'x="16" y="{26 + 22 * i}" font-size="17" '
                       f'font-weight="700" fill="{THEME["accent"]}">{escape(line)}'
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
            fill = '#f7f7f7' if role else 'transparent'
            svg.append(f'<path class="kumo-surface" '
                       f'transform="translate(10 {ry + 2})" '
                       f'd="{_squircle(card_width - 20, rh - 4, 7)}" '
                       f'fill="{fill}"/>')
            for i, line in enumerate(lines):
                svg.append('<text xml:space="preserve" '
                           f'x="18" y="{ry + 21 + 20 * i}" '
                           f'font-size="16" fill="{THEME["foreground"]}">{escape(line)}'
                           '</text>')
            if role:
                svg.append(f'<text x="{card_width - 18}" '
                           f'y="{ry + rh / 2 + 4}" text-anchor="end" '
                           f'font-size="10" font-weight="700" '
                           f'fill="{THEME["accent"]}">{role}</text>')
            svg.append('</g>')
        if footer:
            svg.append(f'<text x="18" y="{card_height - 14}" font-size="12" '
                       f'fill="{THEME["muted"]}">{footer}</text>')
        svg.append('</g>')
    for x, y in geometry['handles']:
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{THEME["muted"]}"/>')
    return SVG(''.join(svg) + '</svg>', samples, uid, layouts=layouts,
               initial_layout=selected)
