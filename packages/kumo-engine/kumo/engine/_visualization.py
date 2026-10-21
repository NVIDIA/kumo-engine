# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free SVG rendering of graph schema metadata."""

from html import escape
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

    def _repr_html_(self) -> str:
        return (
            '<div class="kumo-graph"><style>'
            '.kumo-graph{display:flex;flex-direction:column;max-height:640px;'
            'max-width:100%;padding:8px;box-sizing:border-box;background:white;'
            'color:#202020;font:14px Arial,Helvetica,sans-serif}'
            '.kumo-graph button{align-self:flex-start;padding:6px 12px;'
            'margin-bottom:8px;background:#f4f8ec;color:#386000;'
            'border:1px solid #c8d9ad;border-radius:4px;cursor:pointer}'
            '.kumo-graph-view{overflow:auto;min-height:0}'
            '.kumo-graph-view svg{max-width:none!important;'
            'max-height:none!important}'
            '.kumo-graph:fullscreen{height:100%;max-height:none}'
            '.kumo-graph:fullscreen .kumo-graph-view{flex:1}'
            '.kumo-graph .kumo-graph-exit,.kumo-graph:fullscreen '
            '.kumo-graph-enter{display:none}'
            '.kumo-graph:fullscreen .kumo-graph-exit{display:block}'
            '</style><button type="button" class="kumo-graph-enter" '
            'onclick="this.parentElement.querySelector(\'[role=status]\')'
            '.hidden=true;Promise.resolve().then(() => '
            'this.parentElement.requestFullscreen()).catch(() => '
            'this.parentElement.querySelector(\'[role=status]\').hidden=false)">'
            'Full screen</button><button type="button" class="kumo-graph-exit" '
            'onclick="document.exitFullscreen()">Exit full screen</button>'
            '<span role="status" hidden="">Full screen is unavailable in this '
            'notebook. Scroll here or open the exported SVG.</span>'
            '<div class="kumo-graph-view" tabindex="0" role="region" '
            'aria-label="Scrollable graph visualization">'
            '<div style="width:max-content">' + self + '</div></div>'
            '<script>(()=>{const host=document.currentScript.parentElement;'
            'requestAnimationFrame(()=>host.querySelectorAll('
            '\'.kumo-graph-view:not([data-positioned])\').forEach(v=>{'
            'const p=Array.from(v.querySelectorAll(\'g[transform]\'),'
            'g=>g.transform.baseVal.getItem(0).matrix)'
            '.sort((a,b)=>a.e-b.e||a.f-b.f)[0];'
            'if(p){v.scrollLeft=Math.max(0,p.e-40);'
            'v.scrollTop=Math.max(0,p.f-40)}v.dataset.positioned=true;'
            '}))})()</script></div>')


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
        label = escape(f'{edge.src_table}.{edge.fkey} → '
                       f'{edge.dst_table}.{definitions[edge.dst_table].pkey}')
        svg.append(f'<g><title>{label}</title><path d="{path}" fill="none" '
                   'stroke="#a1a1a1" stroke-width="1"/></g>')
        handles.update(((sx, sy), (dx, dy)))
    for name in names:
        x, y = xy[node_ids[name]]
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
            svg.append('<text xml:space="preserve" '
                       f'x="16" y="{28 + 22 * i}" font-size="17" '
                       f'font-weight="700" fill="#386000">{escape(line)}'
                       '</text>')
        for column, lines, ry, rh in rows:
            role = keys[name].get(column, '')
            if role:
                svg.append(f'<rect x="10" y="{ry + 2}" '
                           f'width="{card_width - 20}" height="{rh - 4}" '
                           'rx="3" fill="#f7f7f7"/>')
            for i, line in enumerate(lines):
                svg.append('<text xml:space="preserve" '
                           f'x="18" y="{ry + 23 + 20 * i}" '
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
    for x, y in sorted(handles):
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#64748b"/>')
    return SVG(''.join(svg) + '</svg>')
