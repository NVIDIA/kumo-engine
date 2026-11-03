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
class Dataset:
    r"""A source table's location, schema, profile and context."""

    name: str
    source: Mapping[str, Optional[str]]
    schema: Mapping[str, Mapping[str, str]]
    profile: Mapping[str, Any]
    context: str


Summary = Union[Scalar, Tag, Table, Note, Dataset]
