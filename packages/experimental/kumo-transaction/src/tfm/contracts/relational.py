# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral, bounded relational input declarations.

Keys and clocks are metadata. Only explicitly mapped fields become inputs.
Relationship versions carry separate effective and knowledge intervals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Literal

from tfm.contracts.schema import ensure_utc


@dataclass(frozen=True)
class RelationalTable:
    """One keyed table with explicit input allowlists and observation clocks.

    Maps are model-field name -> source column. An absent clock is permitted
    only for explicitly immutable rows. Primary keys may be composite.
    """

    name: str
    key: tuple[str, ...]
    numeric: Mapping[str, str] = field(default_factory=dict)
    categorical: Mapping[str, str] = field(default_factory=dict)
    text: Mapping[str, str] = field(default_factory=dict)
    event_time: str | None = None
    available_at: str | None = None
    immutable: bool = False
    role: Literal["input", "labels"] = "input"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", tuple(self.key))
        if not self.name or not self.key or len(set(self.key)) != len(self.key):
            raise ValueError("Relational tables need a name and unique primary-key columns")
        if self.role not in {"input", "labels"}:
            raise ValueError(f"Table {self.name}: unknown table role")
        if self.role == "labels" and (self.numeric or self.categorical or self.text):
            raise ValueError(f"Table {self.name}: label tables cannot declare input features")
        if self.immutable != (self.event_time is None and self.available_at is None):
            raise ValueError(f"Table {self.name}: declare both clocks or immutable=True")
        if not self.immutable and (self.event_time is None or self.available_at is None):
            raise ValueError(f"Table {self.name}: event_time and available_at must be explicit")
        for name in ("numeric", "categorical", "text"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class RelationalForeignKey:
    """A child-to-parent key binding, traversable in either direction.

    ``available_at`` names a child column for the FK observation, or defaults
    to the child's row observation. Versioned bindings are supplied separately
    and replace this fixed binding for the declared role.
    """

    name: str
    parent: str
    child: str
    columns: tuple[str, ...]
    available_at: str | None = None
    ownership: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        if (
            not all(value.strip() for value in (self.name, self.parent, self.child, *self.columns))
            or not self.columns
            or len(set(self.columns)) != len(self.columns)
        ):
            raise ValueError("Foreign keys require nonempty names/endpoints and unique key columns")


@dataclass(frozen=True)
class OwnershipVersion:
    """One relationship fact, effective in [valid_from, valid_until).

    It can be used only in [known_from, known_until) at prediction cutoff.
    Endpoints contain raw primary-key values in their declared column order.
    ``None`` denotes an open interval end, never an unknown observation clock.
    """

    relation: str
    parent_key: tuple[Any, ...]
    child_key: tuple[Any, ...]
    valid_from: datetime
    known_from: datetime
    valid_until: datetime | None = None
    known_until: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_key", tuple(self.parent_key))
        object.__setattr__(self, "child_key", tuple(self.child_key))
        for name in ("valid_from", "known_from", "valid_until", "known_until"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, datetime):
                    raise ValueError(f"Ownership {name} must be a datetime")
                object.__setattr__(self, name, ensure_utc(value))


@dataclass(frozen=True)
class RelationalDataset:
    """One adapter declaration consumed by sequence, graph and hybrid loaders.

    This bounded reference interface snapshots rows; it is not an out-of-core
    storage contract. Optional version rows never become model features.
    """

    tables: tuple[RelationalTable, ...]
    relationships: tuple[RelationalForeignKey, ...]
    rows: Mapping[str, tuple[Mapping[str, Any], ...]]
    ownership_versions: tuple[OwnershipVersion, ...] = ()
    forbidden_fields: frozenset[str] = frozenset()
    observed_through: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("tables", "relationships", "ownership_versions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "forbidden_fields", frozenset(self.forbidden_fields))
        if self.observed_through is not None:
            object.__setattr__(self, "observed_through", ensure_utc(self.observed_through))
        for table, rows in self.rows.items():
            for row in rows:
                for column, value in row.items():
                    if value is not None and not isinstance(
                        value, (str, bool, int, float, date, Decimal)
                    ):
                        raise ValueError(
                            f"{table}.{column}: relational rows require scalar values; normalize source objects in the adapter"
                        )
        object.__setattr__(
            self,
            "rows",
            MappingProxyType(
                {
                    table: tuple(MappingProxyType(dict(row)) for row in rows)
                    for table, rows in self.rows.items()
                }
            ),
        )
