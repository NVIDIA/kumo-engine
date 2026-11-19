# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script-free notebook snapshot of the default Evaluation views."""

import math
from html import escape
from typing import Any

from kumo.engine._explainability import _format as fmt
from kumo.engine._visualization import _wrap
from kumo.engine._widget import THEME, render_static_svg

ACTIVATE = 'To activate the widget, rerun this cell.'
_WIDTH = 448


class Snapshot:
    """One page of summary text, metric values and evaluation charts."""
    def __init__(self, data: dict[str, Any]) -> None:
        self._marks: list[str] = []
        self._text: list[str] = []
        self._y = 12
        self._line(f'{data["model_type"].replace("_", " ").title()} evaluation',
                   size=20, bold=True)
        for value in data['header'].values():
            if value:
                self._line(value)
        cards = next(
            (t['cards'] for t in data['tabs'] if t['id'] == 'evaluation'), [])
        if not cards:
            self._line('No evaluation diagnostics were recorded.')
        for card in cards:
            self._y += 14
            self._marks.append(
                f'<path d="M16 {self._y}h416" stroke="{THEME["grid"]}"/>')
            self._y += 8
            self._line(card['title'], size=17, bold=True)
            view = card['views'][0]
            if view['id'] == 'history':
                count = len(view['experiments'])
                self._line(f'Epoch history · {count} training experiment' +
                           ('s' if count != 1 else ''))
            elif card['id'] == 'metrics':
                self._metrics(view['tiles'])
            elif card['id'] == 'cutoff':
                self._matrix(card)
            elif view['id'] == 'heatmap':
                self._heatmap(view)
            elif card['id'] in ('scores', 'predicted', 'forecast'):
                self._comparison(card['id'], view)
            if card.get('coverage'):
                self._line(card['coverage'])

    def _label(self, text: str, x: int, y: int, size: int = 14,
               anchor: str = 'start', bold: bool = False) -> None:
        self._marks.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}">{escape(text)}</text>')
        self._text.append(text)

    def _line(self, text: str, size: int = 14, bold: bool = False) -> None:
        for line in _wrap(text, int(730 / size)):
            self._y += size + 6
            self._label(line, 16, self._y, size, bold=bold)

    def _metrics(self, tiles: list[dict[str, Any]]) -> None:
        self._y += 23
        self._label('Metric', 16, self._y, bold=True)
        self._label('Holdout', 249, self._y, anchor='end', bold=True)
        scope = 'Training · last epoch' if any(
            t.get('training_scope') for t in tiles) else 'Training'
        self._label(scope, 432, self._y, anchor='end', bold=True)
        for tile in tiles:
            names = _wrap(tile['name'], 22)
            self._y += 23
            self._label(names[0], 16, self._y)
            self._label(tile['holdout_text'], 249, self._y, anchor='end')
            self._label(tile['training_text'] or 'N/A', 432, self._y,
                        anchor='end')
            for name in names[1:]:
                self._line(name)
            if tile.get('baselines'):
                self._line(' · '.join(b['text'] for b in tile['baselines']),
                           size=13)

    def _matrix(self, card: dict[str, Any]) -> None:
        cls = card['classes'][0]
        point = cls['cutoffs'][card['default']]
        self._line(
            f'Confusion matrix at cutoff {card["grid_text"][card["default"]]}')
        self._y += 23
        for j, column in enumerate(cls['matrix_columns']):
            self._label(column, 235 + j * 134, self._y, anchor='middle',
                        bold=True)
        for i, label in enumerate(cls['matrix_rows']):
            self._y += 34
            self._label(label, 16, self._y)
            for j in range(2):
                self._marks.append(
                    f'<rect x="{173 + j * 134}" y="{self._y - 20}" '
                    f'width="124" height="28" rx="4" '
                    f'fill="{THEME["row_highlight" if j == 0 else "surface"]}"/>'
                )
                self._label(point['cells'][i + 2 * j], 235 + j * 134, self._y,
                            anchor='middle')
        self._y += 8
        self._line(point['matrix_callout'])

    def _heatmap(self, view: dict[str, Any]) -> None:
        self._line(view['label'], bold=True)
        self._line(view['y'])
        counts = view['counts']
        rows, columns = len(counts), len(counts[0])
        px, py = 324 / columns, 150 / rows
        self._y += 12
        top, left = self._y, 84
        self._marks.append(
            f'<rect x="{left}" y="{top}" width="324" height="150" '
            f'fill="{THEME["surface"]}"/>')
        peak_row, peak_col = view['densest']
        peak = counts[peak_row][peak_col]
        for r, row in enumerate(counts):
            for c, count in enumerate(row):
                if count:
                    opacity = 0.12 + 0.78 * math.log1p(count) / math.log1p(peak)
                    self._marks.append(
                        f'<rect x="{left + c * px:.3f}" '
                        f'y="{top + (rows - 1 - r) * py:.3f}" '
                        f'width="{px:.3f}" height="{py:.3f}" '
                        f'fill="{THEME["accent"]}" opacity="{opacity:.2f}"/>')
        x_min, x_max = view['x_edges'][0], view['x_edges'][-1]
        y_min, y_max = view['y_edges'][0], view['y_edges'][-1]
        lo, hi = max(x_min, y_min), min(x_max, y_max)
        if lo <= hi:
            segment = [
                f'{left + 324 * (v - x_min) / (x_max - x_min):.2f} '
                f'{top + 150 * (y_max - v) / (y_max - y_min):.2f}'
                for v in (lo, hi)
            ]
            self._marks.append(
                f'<path d="M{segment[0]}L{segment[1]}" fill="none" '
                f'stroke="{THEME["foreground"]}" stroke-dasharray="5 4"/>')
        self._label(fmt.number(y_max), left - 6, top + 8, 13, 'end')
        self._y += 150
        self._label(fmt.number(view['y_edges'][0]), left - 6, self._y, 13,
                    'end')
        self._y += 18
        self._label(fmt.number(view['x_edges'][0]), left, self._y, 13)
        self._label(fmt.number(view['x_edges'][-1]), left + 324, self._y, 13,
                    'end')
        self._line(view['x'])
        if lo <= hi:
            self._line('Dashed line: ' + view['diagonal'])
        self._line(f'Densest cell: {view["densest_callout"]}, '
                   f'{view["x"]} {view["columns"][peak_col]}, '
                   f'{view["y"]} {view["rows"][peak_row]}')
        if view.get('coverage'):
            self._line(view['coverage'])

    def _comparison(self, card: str, view: dict[str, Any]) -> None:
        keys = {
            'scores': ('score', 'observed'),
            'predicted': ('predicted', 'actual'),
            'forecast': ('model', 'zero')
        }[card]
        rows = view.get('tenths', view.get('steps', []))
        self._line(view['label'] + ' · ' + view['y'])
        values = [
            v for row in rows for key in keys
            if isinstance(v := row[key], (int, float)) and math.isfinite(v)
        ]
        low, high = min([0, *values]), max([0, *values])
        colors = (THEME['accent'], THEME['edge'])
        for i, label in enumerate(view['series']):
            self._y += 20
            self._marks.append(f'<rect x="16" y="{self._y - 11}" width="14" '
                               f'height="9" fill="{colors[i]}"/>')
            self._label(label, 36, self._y)
        # Each series uses the same axis, including a shared zero for signed values.
        scale = 230 / (high - low) if high > low else 0
        zero = 112 - low * scale
        for row in rows:
            self._y += 34
            self._label(row['label'], 16, self._y)
            for i, key in enumerate(keys):
                y = self._y - 14 + i * 14
                value = row[key]
                if isinstance(value, (int, float)) and math.isfinite(value):
                    end = zero + value * scale
                    self._marks.append(
                        f'<rect x="{min(end, zero):.2f}" y="{y}" width="{abs(end-zero):.2f}" '
                        f'height="8" fill="{colors[i]}"/>')
                self._label(row[key + '_text'], 430, y + 8, 13, 'end')
            self._marks.append(f'<path d="M{zero:.2f} {self._y - 14}v22" '
                               f'stroke="{THEME["edge"]}"/>')
        if view.get('reference_callout'):
            self._line(view['reference_callout'])

    def svg(self, *, note: bool = True) -> str:
        height = self._y + (46 if note else 16)
        message = (
            f'<text x="16" y="{height - 16}" font-size="14">{ACTIVATE}</text>'
            if note else '')
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" '
            f'height="{height}" viewBox="0 0 {_WIDTH} {height}" '
            f'font-family="{THEME["font"]}" fill="{THEME["foreground"]}" '
            'role="img"><title>Training diagnostics</title>'
            f'<rect width="100%" height="100%" fill="{THEME["background"]}"/>' +
            ''.join(self._marks) + message + '</svg>')

    def html(self) -> str:
        return ('<div class="kx-static">' + render_static_svg(
            self.svg(note=False), ACTIVATE, ' '.join(self._text)) + '</div>')

    def mime(self) -> dict[str, str]:
        return {
            'text/html': self.html(),
            'image/svg+xml': self.svg(),
            'text/plain': self.plain()
        }

    def plain(self) -> str:
        return '\n'.join([*self._text, ACTIVATE])
