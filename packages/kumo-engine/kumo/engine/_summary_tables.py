# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Series and histograms as tables, and summaries routed for trackers."""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from kumo_core.summaries import (
    Confusion,
    Curve,
    Dataset,
    Histogram,
    Note,
    Scalar,
    Series,
    Summary,
    Table,
    Tag,
)

_DEFINITIONS = frozenset({'from', 'to', 'threshold'})


@dataclass
class SummaryTables:
    series: Dict[str, Tuple[str, Dict[str, Dict[int, float]]]] = field(
        default_factory=dict)
    histograms: Dict[str, List[Histogram]] = field(default_factory=dict)

    def add_series(self, summary: Series) -> None:
        parent, _, column = summary.name.rpartition('/')
        x_label, columns = self.series.setdefault(
            parent or summary.name, (summary.x_label, {}))
        columns[column] = dict(summary.points)

    def add_curve(self, summary: Curve) -> None:
        self.add_series(summary.x)
        self.add_series(summary.y)

    def add_histogram(self, summary: Histogram) -> None:
        name = summary.name.rpartition('/')[0] or summary.name
        group = self.histograms.get(name)
        if group and group[0].edges != summary.edges:
            name = summary.name
        self.histograms.setdefault(name, []).append(summary)

    def tables(self) -> List[Table]:
        return [*(_series_table(name, x_label, columns)
                  for name, (x_label, columns) in self.series.items()),
                *(_histogram_table(name, histograms)
                  for name, histograms in self.histograms.items())]


@dataclass
class Routed:
    r"""Summaries sorted by how a tracker writes them."""

    scalars: List[Scalar] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    charts: Dict[str, Tuple[Tuple[int, float], ...]] = field(
        default_factory=dict)
    tables: List[Table] = field(default_factory=list)
    card: List[Summary] = field(default_factory=list)
    datasets: List[Dataset] = field(default_factory=list)


def route(summaries: Sequence[Summary], charts: bool) -> Routed:
    r"""Brief tables and notes go on the card; series, curves and histograms
    become tables; chart series stay charts only when ``charts``."""
    routed, grouped, tables = Routed(), SummaryTables(), []
    for summary in summaries:
        if isinstance(summary, Scalar):
            routed.scalars.append(summary)
        elif isinstance(summary, Tag):
            routed.tags[summary.name] = summary.value
        elif isinstance(summary, Series) and summary.chart and charts:
            routed.charts[summary.name] = summary.points
        elif isinstance(summary, Series):
            grouped.add_series(summary)
        elif isinstance(summary, Curve):
            grouped.add_curve(summary)
        elif isinstance(summary, Histogram):
            grouped.add_histogram(summary)
        elif isinstance(summary, Dataset):
            routed.datasets.append(summary)
        elif isinstance(summary, Note) or (isinstance(summary, Table)
                                           and summary.brief):
            routed.card.append(summary)
        elif isinstance(summary, Confusion):
            tables.append(confusion_table(summary))
        elif isinstance(summary, Table):
            tables.append(summary)
        else:
            raise TypeError(f'No route for {type(summary).__name__}')
    routed.tables = [*grouped.tables(), *tables]
    return routed


def confusion_table(summary: Confusion) -> Table:
    return Table(summary.name, ('actual', *summary.labels), tuple(
        (label, *row) for label, row in zip(summary.labels, summary.counts)))


def _series_table(name: str, x_label: str,
                  columns: Mapping[str, Mapping[int, float]]) -> Table:
    steps = sorted({step for points in columns.values() for step in points})
    return Table(name, (x_label, *columns), tuple(
        (step, *(points.get(step) for points in columns.values()))
        for step in steps))


def _histogram_table(name: str, histograms: Sequence[Histogram]) -> Table:
    r"""Same-edge histograms as one table, empty bins dropped."""
    edges = histograms[0].edges
    extents = [h.extent for h in histograms if h.extent is not None]
    first = min((extent[0] for extent in extents), default=None)
    last = max((extent[1] for extent in extents), default=None)
    columns = [h.name.rpartition('/')[2] for h in histograms]
    shares = []
    for histogram in histograms:
        total, running = sum(histogram.counts), 0.0
        cumulative = []
        for count in histogram.counts:
            running += count
            cumulative.append(running / total if total else None)
        shares.append(cumulative)
    return Table(name, (
        'from', 'to', *columns,
        *(f'{column} cumulative share' for column in columns)), tuple(
            (low, high, *(h.counts[i] for h in histograms),
             *(share[i] for share in shares))
            for i, (low, high) in enumerate(zip((first, *edges),
                                                (*edges, last)))
            if any(h.counts[i] for h in histograms)))


def table_json(table: Table) -> str:
    r"""Table JSON for tracker table views."""
    digits = [10 if column in _DEFINITIONS else 6 for column in table.columns]
    return json.dumps({'columns': list(table.columns), 'data': [
        [_rounded(value, places) for value, places in zip(row, digits)]
        for row in table.rows]}, default=str)


def _rounded(value: Any, digits: int) -> Any:
    if not isinstance(value, float) or not math.isfinite(value) or not value:
        return value
    return round(value, digits - 1 - math.floor(math.log10(abs(value))))
