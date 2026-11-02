# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from xml.etree import ElementTree

from kumo_core.summaries import BarChart

from kumo.engine._svg import bar_chart

_SVG = '{http://www.w3.org/2000/svg}'


def test_a_bar_chart_draws_bars_axes_and_an_optional_line():
    chart = BarChart('pred/s', 'Scores of 6 users <a&b>', ('score', 'users'),
                     (0.0, 0.5, 1.0), (2.0, 4.0), (3.0, 3.0), ('now', 'then'))

    for drawn, line in ((chart, True), (replace(chart, line=()), False)):
        svg = ElementTree.fromstring(bar_chart(drawn))
        texts = [text.text for text in svg.iter(f'{_SVG}text')]
        bars = [rect for rect in svg.iter(f'{_SVG}rect')
                if rect.get('fill') == '#2a78d6']
        assert texts[0] == 'Scores of 6 users <a&b>'
        assert {'0', '0.5', '1', 'score', 'users'} <= set(texts)
        assert len(bars) == 2 + line
        assert len(list(svg.iter(f'{_SVG}polyline'))) == line
        assert ('then' in texts) == line


def test_uneven_bins_get_equal_bars_labelled_by_their_edges():
    chart = BarChart('pred/p', 't', ('predicted', 'users'),
                     (-3.0, 0.0, 1.0, 10.0, 250.0), (1.0, 2.0, 3.0, 4.0))

    svg = ElementTree.fromstring(bar_chart(chart))
    texts = [text.text for text in svg.iter(f'{_SVG}text')]
    widths = {rect.get('width') for rect in svg.iter(f'{_SVG}rect')
              if rect.get('fill') == '#2a78d6'}
    assert {'-3', '0', '1', '10', '250'} <= set(texts)
    assert len(widths) == 1
