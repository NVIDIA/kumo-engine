# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometry invariants for notebook layout alternatives."""
import copy
import math
import re

import pytest

from kumo.engine._visualization_layout import build_layouts, choose_layout


def _card(name, rank=0, height=180, x=80, y=80):
    return dict(name=name, width=304, height=height, rank=rank, x=x, y=y)


def _edge(source, target, port=90):
    return dict(source=source, target=target, source_port=60, target_port=port)


def _standard(cards):
    return dict(width=888, height=3200,
                positions={c['name']: [c['x'], c['y']] for c in cards},
                paths=[], handles=[])


def _samples(path):
    tokens = re.findall(r'[MLQC]|-?\d+(?:\.\d+)?', path)
    i, current = 0, (0, 0)
    while i < len(tokens):
        command = tokens[i]
        i += 1
        count = {'M': 1, 'L': 1, 'Q': 2, 'C': 3}[command]
        points = [current]
        for _ in range(count):
            points.append((float(tokens[i]), float(tokens[i + 1])))
            i += 2
        if command == 'M':
            current = points[-1]
            yield current
            continue
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        for step in range(max(2, math.ceil(length / 5)) + 1):
            t = step / max(2, math.ceil(length / 5))
            work = points[:]
            while len(work) > 1:
                work = [((1 - t) * a[0] + t * b[0],
                         (1 - t) * a[1] + t * b[1])
                        for a, b in zip(work, work[1:])]
            yield work[0]
        current = points[-1]


def _assert_geometry(cards, edges, geometry):
    rectangles = []
    for card in cards:
        x, y = geometry['positions'][card['name']]
        rect = (x, y, x + card['width'], y + card['height'])
        assert 0 <= x < rect[2] <= geometry['width']
        assert 0 <= y < rect[3] <= geometry['height']
        assert all(rect[2] <= r[0] or rect[0] >= r[2]
                   or rect[3] <= r[1] or rect[1] >= r[3] for r in rectangles)
        rectangles.append(rect)
    lookup = {c['name']: c for c in cards}
    assert len(geometry['paths']) == len(edges)
    for edge, path in zip(edges, geometry['paths']):
        points = list(_samples(path))
        sx, sy = geometry['positions'][edge['source']]
        tx, ty = geometry['positions'][edge['target']]
        assert points[0] == pytest.approx(
            (sx + lookup[edge['source']]['width'], sy + edge['source_port']),
            abs=.001)
        assert points[-1] == pytest.approx((tx, ty + edge['target_port']),
                                           abs=.001)
        for x, y in points:
            assert -.001 <= x <= geometry['width'] + .001
            assert -.001 <= y <= geometry['height'] + .001
            assert not any(a + .01 < x < c - .01 and b + .01 < y < d - .01
                           for a, b, c, d in rectangles)


def test_standard_geometry_is_preserved_without_mutating_inputs():
    cards = [_card('root'), _card('leaf', 1)]
    edges = [_edge('root', 'leaf')]
    standard = _standard(cards)
    before = copy.deepcopy((cards, edges, standard))
    layouts = build_layouts(cards, edges, standard)
    assert set(layouts) == {'standard', 'compact'}
    assert layouts['standard'] is standard
    assert (cards, edges, standard) == before


def test_empty_graph_has_valid_alternatives():
    standard = dict(width=304, height=112, positions={}, paths=[], handles=[])
    layouts = build_layouts([], [], standard)
    assert set(layouts) == {'standard', 'compact'}
    assert all(g == standard for g in layouts.values())
    assert choose_layout([], [], layouts) == 'standard'


def test_ring_uses_standard_default_and_has_valid_compact_alternative():
    names = ['zeta', 'alpha', 'omega', 'beta', 'theta', 'gamma', 'rho', 'eta']
    cards = [_card(name, rank=i, height=118 + i * 40)
             for i, name in enumerate(names)]
    edges = [_edge(a, b) for a, b in zip(names, names[1:] + names[:1])]
    layouts = build_layouts(cards, edges, _standard(cards))
    assert set(layouts) == {'standard', 'compact'}
    assert choose_layout(cards, edges, layouts) == 'standard'
    _assert_geometry(cards, edges, layouts['compact'])


def test_crowded_sparse_rank_splits_into_only_one_extra_column():
    cards = [_card('hub-a', height=118), _card('hub-b', height=342, y=300)]
    cards += [_card(f'branch-{i}', 1, height=396 if i % 2 else 182,
                    x=504, y=i * 350) for i in range(7)]
    edges = [_edge(hub, f'branch-{i}', 90 if hub == 'hub-a' else 120)
             for i in range(7) for hub in ('hub-a', 'hub-b')]
    layouts = build_layouts(cards, edges, _standard(cards))
    compact = layouts['compact']
    assert len({p[0] for p in compact['positions'].values()}) == 3
    assert compact['height'] < 2000
    assert choose_layout(cards, edges, layouts) == 'compact'
    _assert_geometry(cards, edges, compact)
    horizontal = []
    for path in compact['paths']:
        points = list(_samples(path))
        spans = {}
        for a, b in zip(points, points[1:]):
            if a[1] == b[1] and a[0] != b[0]:
                low, high = spans.get(a[1], (math.inf, -math.inf))
                spans[a[1]] = (min(low, a[0], b[0]), max(high, a[0], b[0]))
        for other in horizontal:
            for y in spans.keys() & other.keys():
                overlap = min(spans[y][1], other[y][1]) - max(
                    spans[y][0], other[y][0])
                # Shared key ports may share a short horizontal stem.
                assert overlap < 64
        horizontal.append(spans)


def test_layout_and_routes_do_not_depend_on_input_order():
    cards = [_card(f'n-{i}', rank=i % 3, y=i * 100) for i in range(8)]
    edges = [_edge(f'n-{i}', f'n-{(i + 1) % 8}') for i in range(8)]
    edges.append(edges[0].copy())
    standard = _standard(cards)
    first = build_layouts(cards, edges, standard)
    second = build_layouts(cards[::-1], edges[::-1], standard)
    assert first['compact']['positions'] == second['compact']['positions']
    assert first['compact']['paths'] == second['compact']['paths'][::-1]
    assert first['compact']['handles'] == second['compact']['handles']


@pytest.mark.parametrize('shape', ['mesh', 'disconnected', 'many-to-many'])
def test_routes_avoid_cards_across_difficult_shapes(shape):
    cards = [_card(f'table-{i}', rank=i // 4, height=150 + 40 * (i % 4),
                   x=80 + (i // 4) * 424, y=80 + (i % 4) * 400)
             for i in range(12)]
    if shape == 'mesh':
        edges = [_edge(a['name'], b['name'], 90 + 4 * i)
                 for i, a in enumerate(cards) for b in cards if a != b]
    elif shape == 'disconnected':
        edges = [_edge(f'table-{i}', f'table-{i + 1}')
                 for i in (0, 1, 4, 5, 6, 8)]
    else:
        edges = [_edge(f'table-{i}', f'table-{j}', 90 + 4 * i)
                 for i in range(4) for j in range(4, 12)]
    layouts = build_layouts(cards, edges, _standard(cards))
    _assert_geometry(cards, edges, layouts['compact'])
