# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vector graph diagrams with a paginated schema and relationship index."""

from functools import lru_cache
from html import escape
from importlib.resources import files
from io import BytesIO
from itertools import groupby

from lxml import etree as ET
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)
from svglib.fonts import FontMap
from svglib.svglib import svg2rlg

from kumo.engine._widget import THEME


class _PDFFontMap(FontMap):
    def find_font(self, font_name, weight='normal', style='normal'):
        if font_name == 'KumoPDF':
            return ('KumoPDF-Bold' if weight in ('bold', '700')
                    else 'KumoPDF'), True
        if font_name in ('STSong-Light', 'HeiseiKakuGo-W5', 'HYSMyeongJo-Medium'):
            return font_name, True
        return super().find_font(font_name, weight, style)


@lru_cache(maxsize=1)
def _fonts():
    directory = files('matplotlib').joinpath('mpl-data/fonts/ttf')
    for suffix in ('', '-Bold'):
        name = 'KumoPDF' + suffix
        font = TTFont(name, str(directory.joinpath(f'DejaVuSans{suffix}.ttf')))
        pdfmetrics.registerFont(font)
    for name in ('STSong-Light', 'HeiseiKakuGo-W5', 'HYSMyeongJo-Medium'):
        pdfmetrics.registerFont(UnicodeCIDFont(name))
    pdfmetrics.registerFontFamily('KumoPDF', normal='KumoPDF',
                                  bold='KumoPDF-Bold', italic='KumoPDF',
                                  boldItalic='KumoPDF-Bold')
    return _PDFFontMap(), pdfmetrics.getFont('KumoPDF').face.charToGlyph


def _font_runs(text):
    supported = _fonts()[1]

    def font(char):
        code = ord(char)
        if code in supported or char in '\n\t':
            return 'KumoPDF'
        if 0x3400 <= code <= 0x9FFF:
            return 'STSong-Light'
        if 0x3000 <= code <= 0x33FF or 0xFF00 <= code <= 0xFFEF:
            return 'HeiseiKakuGo-W5'
        if 0xAC00 <= code <= 0xD7AF:
            return 'HYSMyeongJo-Medium'
        return None

    for name, chars in groupby(text, font):
        value = ''.join(chars)
        yield name or 'KumoPDF', (value if name else
                                  value.encode('unicode_escape').decode('ascii'))


def render_pdf(svg: str, geometry: dict, index: list[dict]) -> bytes:
    """Export an entire layout and every declared column without sample values."""
    font_map, _ = _fonts()
    escaped = False
    root = ET.fromstring(svg.encode('utf-8'))
    ns = '{http://www.w3.org/2000/svg}'
    width, height = geometry['width'], geometry['height']
    root.set('width', str(width))
    root.set('height', str(height))
    root.set('viewBox', f'0 0 {width} {height}')
    root.set('font-family', 'KumoPDF')
    for child in list(root):
        if child.tag in (ns + 'defs', ns + 'circle') or (
                child.tag == ns + 'rect' and
                child.get('fill', '').startswith('url(')):
            root.remove(child)
        elif child.tag == ns + 'rect':
            child.set('width', str(width))
            child.set('height', str(height))
        elif 'data-kumo-card' in child.attrib:
            x, y = geometry['positions'][child.get('data-kumo-card')]
            child.set('transform', f'translate({x} {y})')
    for path, value in zip(root.findall(f'.//{ns}path[@class="kumo-edge"]'),
                           geometry['paths']):
        path.set('d', value)
    for x, y in geometry['handles']:
        ET.SubElement(root, ns + 'circle', cx=str(x), cy=str(y), r='4',
                      fill=THEME['muted'])
    for element in root.iter():
        if element.get('fill') == 'transparent':
            element.set('fill', 'none')
    for element in root.iter(ns + 'text'):
        original = element.text or ''
        runs = list(_font_runs(original))
        count = sum(len(text) for _, text in runs)
        size = float(element.get('font-size', 16))
        if count > len(original):
            escaped = True
            size *= len(original) / count
            element.set('font-size', str(size))
        widths = [pdfmetrics.stringWidth(
            text, 'KumoPDF-Bold' if font == 'KumoPDF' and
            element.get('font-weight') == '700' else font, size)
            for font, text in runs]
        x = float(element.get('x', 0))
        if element.get('text-anchor') == 'end':
            x -= sum(widths)
        element.set('text-anchor', 'start')
        element.text = None
        for (font, text), run_width in zip(runs, widths):
            ET.SubElement(element, ns + 'tspan', {
                'font-family': font, 'x': str(x)}).text = text
            x += run_width
    drawing = svg2rlg(BytesIO(ET.tostring(root)), font_map=font_map)
    if drawing is None:
        raise ValueError('Could not render the graph as PDF.')
    graph_size = (min(14400, max(842, drawing.width + 64)),
                  min(14400, max(595, drawing.height + 96)))
    output = BytesIO()
    document = BaseDocTemplate(output, pagesize=graph_size, pageCompression=1,
                               title='Kumo graph and schema index',
                               author='Kumo Engine')

    def graph_page(canvas, doc):
        canvas.setFont('KumoPDF-Bold', 16)
        canvas.setFillColor(colors.HexColor(THEME['accent']))
        canvas.drawString(32, graph_size[1] - 32, 'Table relationships')
        scale = min(1, (graph_size[0] - 64) / drawing.width,
                    (graph_size[1] - 96) / drawing.height)
        canvas.saveState()
        canvas.translate((graph_size[0] - drawing.width * scale) / 2,
                         (graph_size[1] - drawing.height * scale) / 2 - 8)
        canvas.scale(scale, scale)
        renderPDF.draw(drawing, canvas, 0, 0)
        canvas.restoreState()
        canvas.setFont('KumoPDF', 9)
        canvas.drawString(32, 20, 'PK: primary key   FK: foreign key   TS: time column')

    def index_page(canvas, doc):
        canvas.setFont('KumoPDF', 9)
        canvas.setFillColor(colors.HexColor(THEME['muted']))
        canvas.drawRightString(A4[0] - 36, 20, f'Page {doc.page}')

    document.addPageTemplates([
        PageTemplate('graph', [Frame(32, 32, graph_size[0] - 64,
                                     graph_size[1] - 64)], onPage=graph_page,
                     pagesize=graph_size),
        PageTemplate('index', [Frame(30, 30, A4[0] - 60, A4[1] - 60)],
                     onPage=index_page, pagesize=A4),
    ])
    normal = ParagraphStyle('body', fontName='KumoPDF', fontSize=9, leading=13,
                            textColor=colors.HexColor(THEME['foreground']),
                            wordWrap='CJK')
    heading = ParagraphStyle('heading', parent=normal, fontName='KumoPDF-Bold',
                             fontSize=14, leading=19, spaceBefore=12,
                             spaceAfter=8, keepWithNext=True)

    def paragraph(value, style=normal):
        nonlocal escaped
        runs = list(_font_runs(value))
        escaped |= ''.join(text for _, text in runs) != value
        return Paragraph(''.join(
            f'<font name="{style.fontName if font == "KumoPDF" else font}">'
            f'{escape(text).replace(" ", "&#32;").replace(chr(10), "<br/>")}</font>'
            for font, text in runs), style)

    story = [Spacer(1, 1), NextPageTemplate('index'), PageBreak(),
             paragraph('Tables and connections', heading),
             paragraph('All declared columns are listed, including columns '
                       'omitted from the diagram. Sample values are excluded.')]
    for table in index:
        story.append(paragraph(table['name'], heading))
        rows = [[paragraph(value) for value in ('Column', 'Data type', 'Role')]]
        rows.extend([paragraph(col['name']), paragraph(col['type']),
                     paragraph(col['role'] or 'Column')]
                    for col in table['columns'])
        if table['columns']:
            story.append(LongTable(
                rows, colWidths=[285, 140, A4[0] - 72 - 425], repeatRows=1,
                splitInRow=1, hAlign='LEFT', style=[
                    ('BACKGROUND', (0, 0), (-1, 0),
                     colors.HexColor(THEME['header_background'])),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('LINEBELOW', (0, 0), (-1, -1), .3,
                     colors.HexColor(THEME['card_border'])),
                ]))
        else:
            story.append(paragraph('No columns'))
        story.append(Spacer(1, 6))
        story.extend(paragraph(connection) for connection in
                     table['connections'] or ['No outgoing connections'])
    if not index:
        story.append(paragraph('No tables'))
    if escaped:
        story.insert(5, paragraph(
            'Characters unavailable in the PDF fonts use Unicode escapes '
            '(\\uXXXX or \\UXXXXXXXX).'))
    document.build(story)
    return output.getvalue()
