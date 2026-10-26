# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic graph geometry, independent of rendering and viewport state.

All coordinates use SVG units. Edges run from a primary-key table's right
port to a foreign-key table's left port. Layouts never read source data.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import TypedDict


class Card(TypedDict):
    name: str
    width: float
    height: float
    rank: int
    x: float
    y: float


class Edge(TypedDict):
    source: str
    target: str
    source_port: float
    target_port: float


class Geometry(TypedDict):
    width: float
    height: float
    positions: dict[str, list[float]]
    paths: list[str]
    handles: list[list[float]]


Point = tuple[float, float]
Rect = tuple[float, float, float, float]
_GAP = 72.0
_COLUMN_GAP = 140.0
_MARGIN = 80.0


def _components(cards: list[Card], edges: list[Edge]) -> list[list[Card]]:
    lookup = {card['name']: card for card in cards}
    neighbors: dict[str, set[str]] = {name: set() for name in lookup}
    for edge in edges:
        neighbors[edge['source']].add(edge['target'])
        neighbors[edge['target']].add(edge['source'])
    pending = set(lookup)
    result = []
    while pending:
        stack = [min(pending)]
        found = set()
        while stack:
            name = stack.pop()
            if name in found:
                continue
            found.add(name)
            stack.extend(sorted(neighbors[name] - found, reverse=True))
        pending -= found
        result.append([lookup[name] for name in sorted(found)])
    return sorted(result, key=lambda group: (-len(group), group[0]['name']))


def _overlap(a: Rect, b: Rect, gap: float = 0) -> bool:
    return (a[0] < b[2] + gap and a[2] + gap > b[0]
            and a[1] < b[3] + gap and a[3] + gap > b[1])


def _bounds(cards: list[Card], positions: dict[str, list[float]]) -> Rect:
    return (min(positions[c['name']][0] for c in cards),
            min(positions[c['name']][1] for c in cards),
            max(positions[c['name']][0] + c['width'] for c in cards),
            max(positions[c['name']][1] + c['height'] for c in cards))


def _compact_component(cards: list[Card], edges: list[Edge]
                       ) -> dict[str, list[float]]:
    ranks: dict[int, list[Card]] = defaultdict(list)
    for card in cards:
        ranks[card['rank']].append(card)
    sparse_hub = len(ranks) <= 2 and len(edges) <= 2 * len(cards)
    for group in ranks.values():
        group.sort(key=lambda c: (c['y'], c['name']))
    # Dense cyclic ranks are artifacts of breaking cycles, not useful layers.
    # A requested compact view may also wrap a long, single-table chain.
    if len(edges) > 2 * len(cards) or len(ranks) > max(4, len(cards) // 2):
        ordered = sorted(cards, key=lambda c: (c['rank'], c['y'], c['name']))
        ranks = {0: ordered}
    tallest = max(c['height'] for c in cards)
    total = sum(c['height'] + _GAP for c in cards)
    candidates = []
    for column_count in range(1, min(len(cards), 8) + 1):
        limit = max(tallest, total / column_count)
        columns: list[list[Card]] = []
        for rank in sorted(ranks):
            group = ranks[rank]
            parts = min(len(group), max(1, math.ceil(
                sum(c['height'] + _GAP for c in group) / (limit + _GAP))))
            if sparse_hub:
                # Splitting one crowded stack into two is enough for sparse
                # hubs; avoid spreading a simple graph across many columns.
                parts = min(parts, 2)
            remaining = list(group)
            for part in range(parts):
                needed = parts - part
                target = sum(c['height'] + _GAP for c in remaining) / needed
                running, best, stop = 0.0, math.inf, 1
                for at, card in enumerate(remaining[:len(remaining) - needed + 1]):
                    running += card['height'] + _GAP
                    if abs(running - target) < best:
                        best, stop = abs(running - target), at + 1
                columns.append(remaining[:stop])
                remaining = remaining[stop:]
        heights = [sum(c['height'] for c in col) + _GAP * (len(col) - 1)
                   for col in columns]
        positions: dict[str, list[float]] = {}
        x = 0.0
        for column, height in zip(columns, heights):
            y = (max(heights) - height) / 2
            for card in column:
                positions[card['name']] = [x, y]
                y += card['height'] + _GAP
            x += max(c['width'] for c in column) + _COLUMN_GAP
        width, height = x - _COLUMN_GAP, max(heights)
        # A fixed reading canvas keeps geometry independent of the notebook.
        # Count and heights determine packing; viewport fit is a viewer task.
        score = max(width / 1.45, height) + .06 * (width + height)
        span = sum(math.hypot(
            positions[e['source']][0] - positions[e['target']][0],
            positions[e['source']][1] + e['source_port']
            - positions[e['target']][1] - e['target_port']) for e in edges)
        score += .06 * span / max(1, len(edges))
        candidates.append((score, column_count, positions))
    return min(candidates, key=lambda item: item[:2])[2]


def _pack_components(cards: list[Card], edges: list[Edge]
                     ) -> dict[str, list[float]]:
    blocks = []
    for group in _components(cards, edges):
        names = {c['name'] for c in group}
        relations = [e for e in edges if e['source'] in names]
        positions = _compact_component(group, relations)
        left, top, right, bottom = _bounds(group, positions)
        positions = {name: [x - left, y - top]
                     for name, (x, y) in positions.items()}
        blocks.append((right - left, bottom - top, positions))
    # Place separate domains as rectangles. Small isolated tables can fill
    # gaps beside an earlier component instead of forcing another full row.
    gap = 96.0
    minimum = max(w for w, _, _ in blocks)
    total = sum(w + gap for w, _, _ in blocks) - gap
    candidates = []
    for steps in range(9):
        limit = minimum + (total - minimum) * steps / 8
        placed: list[Rect] = []
        result: dict[str, list[float]] = {}
        for width, height, positions in blocks:
            xs = sorted({0.0, *(r[2] + gap for r in placed)})
            ys = sorted({0.0, *(r[3] + gap for r in placed)})
            slots = [(y, x) for y in ys for x in xs
                     if x + width <= limit + 1e-6
                     and not any(_overlap((x, y, x + width, y + height),
                                          r, gap) for r in placed)]
            y, x = min(slots)
            placed.append((x, y, x + width, y + height))
            for name, (cx, cy) in positions.items():
                result[name] = [cx + x + _MARGIN, cy + y + _MARGIN]
        width = max(r[2] for r in placed)
        height = max(r[3] for r in placed)
        score = max(width / 1.45, height) + .06 * (width + height)
        candidates.append((score, steps, result))
    return min(candidates, key=lambda item: item[:2])[2]


def _line_hits(a: Point, b: Point, rect: Rect) -> bool:
    """Whether an open segment enters the interior of a rectangle."""
    if (max(a[0], b[0]) <= rect[0] or min(a[0], b[0]) >= rect[2]
            or max(a[1], b[1]) <= rect[1] or min(a[1], b[1]) >= rect[3]):
        return False
    low, high = 0.0, 1.0
    for axis in (0, 1):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-9:
            if a[axis] <= rect[axis] or a[axis] >= rect[axis + 2]:
                return False
        else:
            start = (rect[axis] - a[axis]) / delta
            end = (rect[axis + 2] - a[axis]) / delta
            low, high = max(low, min(start, end)), min(high, max(start, end))
            if low >= high - 1e-9:
                return False
    return high > 0 and low < 1


def _curve_clear(curve: tuple[Point, ...], rect: Rect, depth: int = 7) -> bool:
    box = (min(p[0] for p in curve), min(p[1] for p in curve),
           max(p[0] for p in curve), max(p[1] for p in curve))
    if not _overlap(box, rect):
        return True
    if depth == 0:
        return False
    a, b, c, d = curve
    ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    bc = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
    cd = ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)
    abc = ((ab[0] + bc[0]) / 2, (ab[1] + bc[1]) / 2)
    bcd = ((bc[0] + cd[0]) / 2, (bc[1] + cd[1]) / 2)
    mid = ((abc[0] + bcd[0]) / 2, (abc[1] + bcd[1]) / 2)
    return (_curve_clear((a, ab, abc, mid), rect, depth - 1)
            and _curve_clear((mid, bcd, cd, d), rect, depth - 1))


def _point(point: Point) -> str:
    return f'{point[0]:.3f} {point[1]:.3f}'


def _smooth(points: list[Point]) -> str:
    # Rounding is smaller than obstacle clearance, so curves cannot cut into
    # cards when replacing a visibility-graph corner.
    path = 'M ' + _point(points[0])
    for a, b, c in zip(points, points[1:], points[2:]):
        before, after = math.dist(a, b), math.dist(b, c)
        if min(before, after) < 1e-9:
            continue
        radius = min(12.0, before * .35, after * .35)
        p = (b[0] + (a[0] - b[0]) * radius / before,
             b[1] + (a[1] - b[1]) * radius / before)
        q = (b[0] + (c[0] - b[0]) * radius / after,
             b[1] + (c[1] - b[1]) * radius / after)
        path += f' L {_point(p)} Q {_point(b)} {_point(q)}'
    return path + ' L ' + _point(points[-1])


class _Router:
    """A reusable visibility graph for one arrangement and clearance lane."""

    def __init__(self, rectangles: list[Rect], clearance: float):
        self.rectangles = [(x - clearance, y - clearance,
                            right + clearance, bottom + clearance)
                           for x, y, right, bottom in rectangles]
        self.points = [(x, y) for left, top, right, bottom in self.rectangles
                       for x, y in ((left, top), (right, top),
                                    (right, bottom), (left, bottom))]
        self.links: list[list[tuple[int, float]]] = [
            [] for _ in self.points]
        for i, a in enumerate(self.points):
            for j in range(i):
                b = self.points[j]
                if self.clear(a, b):
                    distance = math.dist(a, b)
                    self.links[i].append((j, distance))
                    self.links[j].append((i, distance))
        self.connectors: dict[Point, list[tuple[int, float]]] = {}

    def clear(self, a: Point, b: Point) -> bool:
        return not any(_line_hits(a, b, r) for r in self.rectangles)

    def connections(self, point: Point) -> list[tuple[int, float]]:
        if point not in self.connectors:
            self.connectors[point] = [
                (i, math.dist(point, p)) for i, p in enumerate(self.points)
                if self.clear(point, p)]
        return self.connectors[point]

    def route(self, start: Point, end: Point) -> list[Point]:
        if self.clear(start, end):
            return [start, end]
        finish = dict(self.connections(end))
        distance = [math.inf] * len(self.points)
        previous: dict[int, int] = {}
        queue = []
        for node, cost in self.connections(start):
            distance[node] = cost
            heapq.heappush(queue, (cost, node))
        best, last = math.inf, None
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            if cost >= best:
                break
            if node in finish and cost + finish[node] < best:
                best, last = cost + finish[node], node
            for neighbor, step in self.links[node]:
                candidate = cost + step
                if candidate < distance[neighbor]:
                    distance[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        if last is None:
            raise ValueError('Unable to route graph relationship around tables.')
        points = [end]
        while last is not None:
            points.append(self.points[last])
            last = previous.get(last)
        points.append(start)
        return points[::-1]


def _geometry(cards: list[Card], edges: list[Edge]) -> Geometry:
    positions = _pack_components(cards, edges)
    lookup = {c['name']: c for c in cards}
    rectangles = [(positions[c['name']][0], positions[c['name']][1],
                   positions[c['name']][0] + c['width'],
                   positions[c['name']][1] + c['height']) for c in cards]
    routers: dict[int, _Router] = {}
    paths, handles = [], set()
    edge_keys = sorted({(e['source'], e['target'], e['source_port'],
                         e['target_port']) for e in edges})
    lanes = {key: index % 7 for index, key in enumerate(edge_keys)}
    for edge in edges:
        source, target = edge['source'], edge['target']
        sx, sy = positions[source]
        tx, ty = positions[target]
        start = (sx + lookup[source]['width'], sy + edge['source_port'])
        end = (tx, ty + edge['target_port'])
        handles.update((start, end))
        half = (end[0] - start[0]) / 2
        curve = (start, (start[0] + half, start[1]),
                 (end[0] - half, end[1]), end)
        if half >= 24 and all(
                _curve_clear(curve, (rect[0] - 10, rect[1] - 10,
                                     rect[2] + 10, rect[3] + 10))
                for c, rect in zip(cards, rectangles)
                if c['name'] not in (source, target)):
            paths.append('M ' + _point(start) + ' C '
                         + ' '.join(_point(p) for p in curve[1:]))
        else:
            lane = lanes[(source, target, edge['source_port'],
                          edge['target_port'])]
            if lane not in routers:
                routers[lane] = _Router(rectangles, 14 + lane * 3.5)
            offset = 40 + lane * 3.5
            route = routers[lane].route(
                (start[0] + offset, start[1]),
                (end[0] - offset, end[1]))
            paths.append(_smooth([start, *route, end]))
    _, _, right, bottom = _bounds(cards, positions)
    return {'width': right + _MARGIN, 'height': bottom + _MARGIN,
            'positions': positions, 'paths': paths,
            'handles': [list(p) for p in sorted(handles)]}


def build_layouts(cards: list[Card], edges: list[Edge],
                  original: Geometry) -> dict[str, Geometry]:
    """Build alternatives once; callers own camera and layout selection.

    The original geometry is passed through unchanged. Input collections are
    never mutated, and alternate geometry does not depend on viewport size.
    """
    if not cards:
        return dict.fromkeys(('original', 'compact'), original)
    ordered = sorted(cards, key=lambda c: c['name'])
    return {'original': original, 'compact': _geometry(ordered, edges)}


def choose_layout(cards: list[Card], edges: list[Edge],
                  layouts: dict[str, Geometry]) -> str:
    """Conservative structural defaults, selected once when rendering."""
    if not cards:
        return 'original'
    original, compact = layouts['original'], layouts['compact']
    components = _components(cards, edges)
    ranks: dict[int, int] = defaultdict(int)
    for card in cards:
        ranks[card['rank']] += 1
    crowded = max(ranks.values()) >= 7 and original['height'] > 1800
    rank_by_name = {c['name']: c['rank'] for c in cards}
    feedback = any(rank_by_name[e['source']] >= rank_by_name[e['target']]
                   for e in edges)
    dense = len(edges) > 2 * len(cards) and feedback
    # Keep layered many-to-many dimensions in their familiar two-column
    # overview; compact remains available as an explicit alternative.
    if len(ranks) == 2 and min(ranks.values()) >= 7:
        return 'original'
    disconnected = len(components) > 1
    old_extent = max(original['width'] / 1.45, original['height'])
    new_extent = max(compact['width'] / 1.45, compact['height'])
    improvement = 1.0 if disconnected else .85
    if ((crowded or dense or disconnected)
            and new_extent < old_extent * improvement):
        return 'compact'
    return 'original'
