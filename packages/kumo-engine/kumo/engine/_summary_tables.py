# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Summaries routed for trackers, and tables as tracker JSON."""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from kumo_core.summaries import Dataset, Note, Scalar, Summary, Table, Tag

_DEFINITIONS = frozenset({'from', 'to', 'threshold'})


@dataclass
class Routed:
    r"""Summaries sorted by how a tracker writes them."""

    scalars: List[Scalar] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    tables: List[Table] = field(default_factory=list)
    card: List[Summary] = field(default_factory=list)
    datasets: List[Dataset] = field(default_factory=list)


def route(summaries: Sequence[Summary]) -> Routed:
    r"""Sorts summaries by how a tracker writes them; skips unknown types."""
    routed, tables = Routed(), []
    for summary in summaries:
        if isinstance(summary, Scalar):
            routed.scalars.append(summary)
        elif isinstance(summary, Tag):
            routed.tags[summary.name] = summary.value
        elif isinstance(summary, Dataset):
            routed.datasets.append(summary)
        elif isinstance(summary, Note) or (isinstance(summary, Table)
                                           and summary.brief):
            routed.card.append(summary)
        elif isinstance(summary, Table):
            tables.append(summary)
    routed.tables = tables
    return routed


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
