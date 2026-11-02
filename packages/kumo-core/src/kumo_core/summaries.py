# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Typed, tracker-neutral results, named ``<section>/<name>``."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class Scalar:
    r"""A metric; ``role='model'`` also logs it on the model."""

    name: str
    value: float
    role: Optional[Literal['model']] = None


@dataclass(frozen=True)
class Tag:
    name: str
    value: str


@dataclass(frozen=True)
class Series:
    r"""``(step, value)`` points, drawn as a chart where ``chart`` is set."""

    name: str
    points: Tuple[Tuple[int, float], ...]
    x_label: str = 'step'
    chart: bool = False


@dataclass(frozen=True)
class Histogram:
    r"""Counts on ``edges``, with open end bins; ``extent`` is (min, max)."""

    name: str
    edges: Tuple[float, ...]
    counts: Tuple[float, ...]
    extent: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class BarChart:
    r"""Bars of ``counts`` on ``edges``, and an optional ``line`` over them."""

    name: str
    title: str
    axes: Tuple[str, str]
    edges: Tuple[float, ...]
    counts: Tuple[float, ...]
    line: Tuple[float, ...] = ()
    legend: Tuple[str, str] = ('', '')


@dataclass(frozen=True)
class Table:
    r"""A table; a ``brief`` one (at most 12 x 5) goes on the run card."""

    name: str
    columns: Tuple[str, ...]
    rows: Tuple[Tuple[Any, ...], ...]
    brief: bool = False


@dataclass(frozen=True)
class Note:
    r"""A line of text for the run card, under ``section``."""

    section: str
    text: str


@dataclass(frozen=True)
class Confusion:
    r"""``counts[i][j]``: rows of actual ``labels[i]``, predicted ``labels[j]``."""

    name: str
    labels: Tuple[str, ...]
    counts: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class Dataset:
    r"""A source table's location, schema, profile and context."""

    name: str
    source: Mapping[str, Optional[str]]
    schema: Mapping[str, Mapping[str, str]]
    profile: Mapping[str, Any]
    context: str


Summary = Union[Scalar, Tag, Series, Histogram, BarChart, Table, Confusion,
                Note, Dataset]
