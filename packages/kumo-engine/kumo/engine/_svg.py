# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Bar charts as SVG images, one equal-width bar per bin."""

import math
from html import escape
from typing import List

from kumo_core.summaries import BarChart

from kumo.engine._text import number

_WIDTH, _HEIGHT, _LEFT, _RIGHT, _TOP, _BOTTOM = 720, 400, 72, 32, 72, 48
_BARS, _LINE, _INK, _MUTED, _GRID = (
    '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df')


def bar_chart(chart: BarChart) -> str:
    width = _WIDTH - _LEFT - _RIGHT
    height = _HEIGHT - _TOP - _BOTTOM
    bins = len(chart.counts)
    step = _step(max(*chart.counts, *chart.line, 1.0))
    peak = 4 * step

    def x(edge: float) -> float:
        return _LEFT + edge / bins * width

    def y(value: float) -> float:
        return _TOP + height * (1 - value / peak)

    parts: List[str] = [
        f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="#fcfcfb"/>',
        _text(_LEFT, 28, chart.title, 'start', _INK, 15)]
    for tick in range(5):
        level = y(step * tick)
        parts += [
            f'<line x1="{_LEFT}" x2="{_LEFT + width}" y1="{level:.1f}" '
            f'y2="{level:.1f}" stroke="{_GRID}"/>',
            _text(_LEFT - 8, level + 4, number(step * tick), 'end')]
    parts += [_text(x(edge), _TOP + height + 18, number(chart.edges[edge]),
                    'middle')
              for edge in sorted({round(bins * tick / 4) for tick in range(5)})]
    parts += [
        f'<rect x="{x(index) + 0.5:.1f}" y="{y(count):.1f}" '
        f'width="{max(x(1) - x(0) - 1, 0.5):.1f}" '
        f'height="{_TOP + height - y(count):.1f}" fill="{_BARS}"/>'
        for index, count in enumerate(chart.counts) if count]
    if chart.line:
        points = ' '.join(f'{x(index + 0.5):.1f},{y(value):.1f}'
                          for index, value in enumerate(chart.line))
        parts.append(f'<polyline points="{points}" fill="none" '
                     f'stroke="{_LINE}" stroke-width="2"/>')
        left = _LEFT
        for label, color in zip(chart.legend, (_BARS, _LINE)):
            parts += [f'<rect x="{left}" y="42" width="10" height="10" '
                      f'fill="{color}"/>', _text(left + 16, 51, label)]
            left += 40 + 7 * len(label)
    parts += [_text(_LEFT + width / 2, _HEIGHT - 8, chart.axes[0], 'middle'),
              f'<text transform="translate(16 {_TOP + height / 2}) '
              f'rotate(-90)" text-anchor="middle" fill="{_MUTED}" '
              f'font-size="12">{escape(chart.axes[1])}</text>']
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" '
            f'height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
            'font-family="sans-serif">' + ''.join(parts) + '</svg>\n')


def _step(peak: float) -> float:
    unit = 10 ** math.floor(math.log10(peak / 4))
    return next(unit * m for m in (1, 2, 5, 10) if unit * m * 4 >= peak)


def _text(x: float, y: float, text: str, anchor: str = 'start',
          color: str = _MUTED, size: int = 12) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'fill="{color}" font-size="{size}">{escape(text)}</text>')
