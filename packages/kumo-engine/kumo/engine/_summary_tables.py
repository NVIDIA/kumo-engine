# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Series and histograms as the tables every tracker logs: the parent path
of a summary's name names its table, the last segment its column."""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

from kumo_core.summaries import Curve, Histogram, Series, Table


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


def _series_table(name: str, x_label: str,
                  columns: Mapping[str, Mapping[int, float]]) -> Table:
    r"""Series ``<name>/<column>`` as one table: the step, then a column
    per series, a row per step."""
    steps = sorted({step for points in columns.values() for step in points})
    return Table(name, (x_label, *columns), tuple(
        (step, *(points.get(step) for points in columns.values()))
        for step in steps))


def _histogram_table(name: str, histograms: Sequence[Histogram]) -> Table:
    r"""Histograms ``<name>/<column>`` over the same edges as one table: a
    bin per row from ``from`` to ``to`` (open at both ends), a count and a
    cumulative share per histogram."""
    edges = histograms[0].edges
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
            for i, (low, high) in enumerate(zip((None, *edges),
                                                (*edges, None)))))
