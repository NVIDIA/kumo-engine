# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Small typed results of an evaluation or a data summary, named
``<section>/<name>`` (``test/roc``, ``pred/True_PROB/hist``, ``graph/rows``).

They know nothing about experiment trackers: each tracker maps these types to
its own calls, so another rendering (a figure, a UI) is one more mapping.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class Scalar:
    name: str
    value: float


@dataclass(frozen=True)
class Tag:
    name: str
    value: str


@dataclass(frozen=True)
class Series:
    r"""``(step, value)`` points; ``x_label`` says what a step counts."""

    name: str
    points: Tuple[Tuple[int, float], ...]
    x_label: str = 'step'


@dataclass(frozen=True)
class Curve:
    r"""``y`` against ``x``, two series over the same steps, such as ROC's
    true-positive against false-positive rate over thresholds."""

    name: str
    x: Series
    y: Series


@dataclass(frozen=True)
class Histogram:
    r"""``counts[i]`` values in ``[edges[i - 1], edges[i])``: ``counts[0]``
    below ``edges[0]`` and ``counts[-1]`` at or above ``edges[-1]``."""

    name: str
    edges: Tuple[float, ...]
    counts: Tuple[float, ...]


@dataclass(frozen=True)
class Table:
    name: str
    columns: Tuple[str, ...]
    rows: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class Confusion:
    r"""``counts[i][j]`` rows of actual ``labels[i]`` predicted
    ``labels[j]``."""

    name: str
    labels: Tuple[str, ...]
    counts: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class Dataset:
    r"""A source table: ``source`` its ``connector``, ``location``,
    ``version`` and ``version_kind``; ``schema`` each column's ``stype`` and
    ``dtype``; ``profile`` its rows and column statistics; ``context``
    ``training`` or ``scoring``."""

    name: str
    source: Mapping[str, Optional[str]]
    schema: Mapping[str, Mapping[str, str]]
    profile: Mapping[str, Any]
    context: str


Summary = Union[Scalar, Tag, Series, Curve, Histogram, Table, Confusion,
                Dataset]
