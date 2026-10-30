# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vector graph diagrams with a paginated schema and relationship index."""

import re
from functools import lru_cache
from io import BytesIO
from xml.etree import ElementTree as ET

from matplotlib import get_data_path, rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, get_font
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
from matplotlib.transforms import Affine2D

from kumo.engine._widget import THEME


@lru_cache(maxsize=1)
def _fonts():
    directory = get_data_path() + '/fonts/ttf/DejaVuSans'
    paths = [directory + suffix + '.ttf' for suffix in ('', '-Bold')]
    fonts = [FontProperties(fname=path) for path in paths]
    supported = set.intersection(*(set(get_font(path).get_charmap())
                                   for path in paths))
    return fonts, supported


def _path(value):
    """Read only the absolute path commands emitted by our graph layouts."""
    tokens = iter(
        re.findall(r'[MLHVQCZ]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', value))
    vertices, codes = [], []
    x = y = 0
    for command in tokens:
        if command == 'Z':
            vertices.append((0, 0))
            codes.append(Path.CLOSEPOLY)
            continue
        count = {'M': 2, 'L': 2, 'H': 1, 'V': 1, 'Q': 4, 'C': 6}[command]
        points = [float(next(tokens)) for _ in range(count)]
        if command == 'H':
            points = [points[0], y]
        elif command == 'V':
            points = [x, points[0]]
        vertices.extend(zip(points[::2], points[1::2]))
        codes.extend([{
            'M': Path.MOVETO,
            'Q': Path.CURVE3,
            'C': Path.CURVE4
        }.get(command, Path.LINETO)] * (len(points) // 2))
        x, y = points[-2:]
    return Path(vertices, codes)


def render_pdf(svg: str, geometry: dict, index: list[dict]) -> bytes:
    """Export our internally generated SVG subset, without sample metadata."""
    fonts, supported = _fonts()
    metrics = TextToPath()
    escaped = False

    def safe(value):
        nonlocal escaped
        result = ''.join(
            char if ord(char) in supported or char in '\n\t' else char.
            encode('unicode_escape').decode('ascii') for char in value)
        escaped |= result != value
        return result

    def width(value, size: float = 9, bold=False):
        font = fonts[bold].copy()
        font.set_size(size)
        return metrics.get_text_width_height_descent(value, font, False)[0]

    def text(axes, x, y, value, size: float = 9, bold=False, color=None,
             align='left'):
        axes.text(x, y, value, fontproperties=fonts[bold], fontsize=size,
                  color=color or THEME['foreground'], ha=align, va='baseline',
                  parse_math=False, usetex=False, clip_on=False, zorder=3)

    def page(w, h):
        figure = Figure(figsize=(w / 72, h / 72))
        axes = figure.add_axes((0, 0, 1, 1), xlim=(0, w), ylim=(h, 0))
        axes.set_axis_off()
        return figure, axes

    output = BytesIO()
    with rc_context({
            'pdf.fonttype': 42, 'figure.facecolor': THEME['background'],
            'savefig.facecolor': THEME['background'],
            'savefig.edgecolor': THEME['background'],
            'savefig.transparent': False,
    }), PdfPages(
            output, metadata={
                'Title': 'Kumo graph and schema index',
                'Author': 'Kumo Engine'
            }) as pdf:
        w, h = geometry['width'], geometry['height']
        pw, ph = min(14400, max(842, w + 64)), min(14400, max(595, h + 96))
        figure, axes = page(pw, ph)
        text(axes, 32, 32, 'Table relationships', 16, True, THEME['accent'])
        text(axes, 32, ph - 20,
             'PK: primary key   FK: foreign key   TS: time column')
        scale = min(1, (pw - 64) / w, (ph - 96) / h)
        origin = ((pw - w * scale) / 2, (ph - h * scale) / 2 + 8)
        root = ET.fromstring(svg)
        clips = {
            node.get('id'): _path(next(iter(node)).get('d'))
            for node in root.iter() if node.tag.endswith('}clipPath')
        }
        edges = iter(geometry['paths'])

        def draw(node, x: float = 0, y: float = 0):
            tag = node.tag.rsplit('}', 1)[-1]
            if tag in ('defs', 'title', 'desc', 'circle', 'rect'):
                return
            if 'data-kumo-card' in node.attrib:
                x, y = geometry['positions'][node.get('data-kumo-card')]
            elif node.get('transform'):
                dx, dy = map(float,
                             re.findall(r'[-+]?[\d.]+', node.get('transform')))
                x, y = x + dx, y + dy
            transform = Affine2D().translate(
                x, y).scale(scale).translate(*origin) + axes.transData
            if tag == 'path':
                value = next(edges) if node.get(
                    'class') == 'kumo-edge' else node.get('d')
                fill = node.get('fill', 'none')
                patch = PathPatch(
                    _path(value), transform=transform,
                    facecolor='none' if fill == 'transparent' else fill,
                    edgecolor=node.get('stroke', 'none'),
                    linewidth=float(node.get('stroke-width', 1)) * scale)
                if node.get('clip-path'):
                    patch.set_clip_path(clips[node.get('clip-path')[5:-1]],
                                        transform)
                axes.add_patch(patch)
            elif tag == 'text':
                original = node.text or ''
                value = safe(original).expandtabs(4)
                size = float(node.get('font-size', 16))
                bold = node.get('font-weight') == '700'
                if value != original:
                    limit = 210 if size == 16 else 272  # Leave room for PK/FK/TS.
                    size *= min(1, limit / max(1, width(value, size, bold)))
                text(axes, origin[0] + (x + float(node.get('x', 0))) * scale,
                     origin[1] + (y + float(node.get('y', 0))) * scale, value,
                     size * scale, bold, node.get('fill'),
                     'right' if node.get('text-anchor') == 'end' else 'left')
            for child in node:
                draw(child, x, y)

        draw(root)
        for x, y in geometry['handles']:
            axes.add_patch(
                Circle((origin[0] + x * scale, origin[1] + y * scale),
                       4 * scale, color=THEME['muted'], linewidth=0))
        pdf.savefig(figure)
        figure.clear()
        # Index pages use point coordinates so line lengths and page breaks are
        # independent of graph size. Split even a single oversized column row.
        pw, ph = 595.28, 841.89
        figure, axes = page(pw, ph)
        y = 40
        page_number = 2
        columns = (36, 321, 461, pw - 36)

        def next_page():
            nonlocal figure, axes, y, page_number
            text(axes, pw - 36, ph - 20, f'Page {page_number}', align='right',
                 color=THEME['muted'])
            pdf.savefig(figure)
            figure.clear()
            figure, axes = page(pw, ph)
            y = 40
            page_number += 1

        def wrap(value, limit, size=9, bold=False):
            lines = []
            for part in safe(value).expandtabs(4).split('\n'):
                if not part:
                    lines.append('')
                while part:
                    if width(part, size, bold) <= limit:
                        lines.append(part)
                        break
                    lo, hi = 1, len(part)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if width(part[:mid], size, bold) <= limit:
                            lo = mid
                        else:
                            hi = mid - 1
                    cut = lo
                    if cut < len(part) and ' ' in part[:cut]:
                        cut = part.rfind(' ', 0, cut) + 1
                    lines.append(part[:cut])
                    part = part[cut:]
            return lines

        def paragraph(value, size=9, bold=False):
            nonlocal y
            for line in wrap(value, pw - 72, size, bold):
                if y + size + 5 > ph - 40:
                    next_page()
                text(axes, 36, y + size, line, size, bold)
                y += size + 5

        def row(values, header=False):
            nonlocal y
            lines = [
                wrap(value, columns[i + 1] - columns[i] - 12, bold=header)
                for i, value in enumerate(values)
            ]
            offset, count = 0, max(map(len, lines))
            while offset < count:
                if y + 25 > ph - 40:
                    next_page()
                    if not header:
                        row(('Column', 'Data type', 'Role'), True)
                take = min(count - offset, max(1, int(
                    (ph - 40 - y - 12) // 13)))
                height = take * 13 + 12
                axes.add_patch(
                    Rectangle((36, y), pw - 72, height,
                              facecolor=THEME['header_background']
                              if header else 'none', edgecolor='none'))
                for i, cell in enumerate(lines):
                    for n, line in enumerate(cell[offset:offset + take]):
                        text(axes, columns[i] + 6, y + 15 + n * 13, line,
                             bold=header)
                y += height
                axes.plot((36, pw - 36), (y, y), color=THEME['card_border'],
                          linewidth=.3)
                offset += take

        paragraph('Tables and connections', 14, True)
        paragraph(
            'All declared columns are listed, including columns omitted from the diagram. Sample values are excluded.'
        )
        # Check the complete index before deciding whether its fallback note is needed.
        for table in index:
            safe(table['name'] + ''.join(table['connections']))
            for col in table['columns']:
                safe(col['name'] + col['type'] + col['role'])
        if escaped:
            paragraph(
                'Characters unavailable in the PDF fonts use Unicode escapes (\\uXXXX or \\UXXXXXXXX).'
            )
        for table in index:
            y += 12
            if y + 70 > ph - 40:
                next_page()
            paragraph(table['name'], 14, True)
            y += 6
            if table['columns']:
                row(('Column', 'Data type', 'Role'), True)
                for col in table['columns']:
                    row((col['name'], col['type'], col['role'] or 'Column'))
            else:
                paragraph('No columns')
            y += 6
            for connection in table['connections'] or [
                    'No outgoing connections'
            ]:
                paragraph(connection)
        if not index:
            paragraph('No tables')
        text(axes, pw - 36, ph - 20, f'Page {page_number}', align='right',
             color=THEME['muted'])
        pdf.savefig(figure)
        figure.clear()
    return output.getvalue()
