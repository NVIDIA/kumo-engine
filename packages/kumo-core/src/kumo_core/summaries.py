# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Typed, tracker-neutral results, named ``<section>/<name>``."""

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
class Table:
    name: str
    columns: Tuple[str, ...]
    rows: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class Dataset:
    r"""A source table's location, schema, profile and context."""

    name: str
    source: Mapping[str, Optional[str]]
    schema: Mapping[str, Mapping[str, str]]
    profile: Mapping[str, Any]
    context: str


Summary = Union[Scalar, Tag, Table, Dataset]
