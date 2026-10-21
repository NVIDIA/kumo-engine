# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free SVG rendering of graph schema metadata."""

from collections import deque
from html import escape
from math import ceil, sqrt
from typing import Mapping, Sequence
from unicodedata import combining

from kumoapi.graph import Edge, validate_edge_endpoints
from kumoapi.table import TableDefinition

from kumo.engine.table import Table

_COLUMN_LIMIT = 10
_LARGE_GRAPH_COLUMN_LIMIT = 5


class SVG(str):
    """SVG text that displays inline in notebooks."""
    def _repr_svg_(self) -> str:
        return str(self)


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


def render(tables: Mapping[str, Table], edges: Sequence[Edge], *,
           show_cols: bool) -> SVG:
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

    # Breadth-first groups keep related tables close, with a bounded grid
    # rather than an arbitrarily wide chain or tall star.
    order = []
    remaining = set(names)
    while remaining:
        root = min(remaining, key=lambda n: (-len(neighbors[n]), n))
        queue = deque([root])
        remaining.remove(root)
        while queue:
            name = queue.popleft()
            order.append(name)
            for neighbor in sorted(neighbors[name] & remaining):
                remaining.remove(neighbor)
                queue.append(neighbor)
    ncols = max(1, ceil(sqrt(len(names))))
    nrows = max(1, ceil(len(names) / ncols))
    positions = {name: divmod(index, nrows) for index, name in enumerate(order)}
    cards = {}
    for name in names:
        definition = definitions[name]
        pkey = definition.pkey
        columns = sorted((col.name for col in definition.cols), key=lambda col:
                         (col != pkey, col not in keys[name], col))
        limit = (_COLUMN_LIMIT
                 if len(names) <= 10 else _LARGE_GRAPH_COLUMN_LIMIT)
        columns = columns[:max(len(keys[name]), limit if show_cols else 0)]
        title = _wrap(name, 27)
        header = 24 + 22 * len(title)
        rows = []
        ports = {}
        y = header + 8
        for column in columns:
            lines = _wrap(column, 22)
            height = max(36, 16 + 20 * len(lines))
            ports[column] = y + height / 2
            rows.append((column, lines, y, height))
            y += height
        hidden = len(definition.cols) - len(columns)
        footer = (f'+ {hidden} other column' +
                  ('s' if hidden != 1 else '') if hidden else
                  ('No columns' if not columns else ''))
        height = y + (30 if footer else 8)
        cards[name] = (title, header, rows, ports, footer, height)

    card_width = 304
    gap = 90 + 8 * len(edges)
    row_heights = [
        max((cards[name][-1]
             for name in names if positions[name][1] == row), default=90)
        for row in range(nrows)
    ]
    distant = [
        edge for edge in edges
        if abs(positions[edge.src_table][0] - positions[edge.dst_table][0]) > 1
    ]
    top = 40 + 12 * len(distant)
    xy = {
        name:
        (40 + col * (card_width + gap), top + sum(row_heights[:row]) + row * 48)
        for name, (col, row) in positions.items()
    }
    width = (ncols * card_width + (ncols - 1) * gap + 80 + (gap if any(
        positions[e.src_table][0] == positions[e.dst_table][0] == ncols - 1
        for e in edges) else 0))
    height = top + sum(row_heights) + (nrows - 1) * 48 + 40
    if not names:
        width, height = 304, 112
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Table relationships" '
        'font-family="Arial, Helvetica, sans-serif">',
        '<title>Table relationships</title>',
        '<desc>Lines connect foreign keys (FK) to primary keys (PK).'
        '</desc>', f'<rect width="{width}" height="{height}" fill="white"/>'
    ]
    if not names:
        svg.append('<text x="24" y="60" font-size="16" fill="#606060">'
                   'No tables</text>')
    lane = 0
    for index, edge in enumerate(edges):
        src, dst = edge.src_table, edge.dst_table
        sx, sy = xy[src]
        dx, dy = xy[dst]
        sy += cards[src][3][edge.fkey]
        dy += cards[dst][3][definitions[dst].pkey]
        sc, dc = positions[src][0], positions[dst][0]
        source_side = 1 if sc <= dc else -1
        destination_side = 1 if sc >= dc else -1
        sx += card_width if source_side == 1 else 0
        dx += card_width if destination_side == 1 else 0
        start, end = (sx, sy), (dx, dy)
        # Opposing ports use distinct horizontal levels so unrelated edges
        # on aligned rows do not appear to join across a gutter.
        sx, sy = sx + source_side * 12, sy + source_side * 6
        dx, dy = dx + destination_side * 12, dy + destination_side * 6
        if sc == dc:
            bend = sx + 24 + index * 8
            path = f'H {bend} V {dy} H {dx}'
        else:
            direction = 1 if sc < dc else -1
            if abs(sc - dc) == 1:
                bend = (sx + dx) / 2 + direction * (index * 8 - len(edges) * 4)
                path = (f'H {bend - direction * 8} '
                        f'Q {bend} {sy} {bend} {sy + (8 if dy > sy else -8)} '
                        f'V {dy - (8 if dy > sy else -8)} '
                        f'Q {bend} {dy} {bend + direction * 8} {dy} H {dx}')
                if sy == dy:
                    path = f'H {dx}'
            else:
                overhead = 24 + lane * 12
                lane += 1
                a = sx + direction * (24 + index * 8)
                b = dx - direction * (24 + index * 8)
                path = (f'H {a} V {overhead} H {b} '
                        f'V {dy} H {dx}')
        path = (f'M {start[0]} {start[1]} L {sx} {sy} {path} '
                f'L {end[0]} {end[1]}')
        label = escape(f'{src}.{edge.fkey} → '
                       f'{dst}.{definitions[dst].pkey}')
        svg.append(f'<g><title>{label}</title><path d="{path}" fill="none" '
                   'stroke="#76b900" stroke-width="1.7" '
                   'stroke-linejoin="round"/></g>')
        for x, y in (start, end):
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" '
                       'fill="#76b900" stroke="white" stroke-width="1.5"/>')
    for name in names:
        x, y = xy[name]
        title, header, rows, ports, footer, card_height = cards[name]
        svg.append(f'<g transform="translate({x} {y})">'
                   f'<title>{escape(name)}</title>'
                   f'<rect width="{card_width}" height="{card_height}" '
                   'rx="4" fill="white" stroke="#94a3b8"/>')
        svg.append(f'<path d="M 4 0 H {card_width - 4} Q {card_width} 0 '
                   f'{card_width} 4 V {header} H 0 V 4 Q 0 0 4 0" '
                   'fill="#f4f8ec"/>'
                   f'<path d="M 0 {header} H {card_width}" '
                   'stroke="#c8d9ad"/>')
        for i, line in enumerate(title):
            svg.append(f'<text x="16" y="{28 + 22 * i}" font-size="17" '
                       f'font-weight="700" fill="#386000">{escape(line)}'
                       '</text>')
        for column, lines, ry, rh in rows:
            role = keys[name].get(column, '')
            if role:
                svg.append(f'<rect x="10" y="{ry + 2}" '
                           f'width="{card_width - 20}" height="{rh - 4}" '
                           'rx="3" fill="#f7f7f7"/>')
            for i, line in enumerate(lines):
                svg.append(f'<text x="18" y="{ry + 23 + 20 * i}" '
                           f'font-size="16" fill="#202020">{escape(line)}'
                           '</text>')
            if role:
                svg.append(f'<text x="{card_width - 18}" '
                           f'y="{ry + rh / 2 + 4}" text-anchor="end" '
                           f'font-size="10" font-weight="700" '
                           f'fill="#386000">{role}</text>')
        if footer:
            svg.append(f'<text x="18" y="{card_height - 14}" font-size="12" '
                       f'fill="#64748b">{footer}</text>')
        svg.append('</g>')
    return SVG(''.join(svg) + '</svg>')
