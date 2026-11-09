# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the relational data means: tables, their keys, clocks, and joins.

A schema is independent of any query, model, engine, or physical location. It
says what the rows *are*; `kumo.pytorch.dataset` says where their bytes live and
`kumo.pytorch.engine` says what reads them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from kumo.pytorch.contract._frozen import FrozenMapping
from kumo.pytorch.contract.dtypes import DType


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return FrozenMapping(dict(value))


__all__ = [
    "NON_TEMPORAL",
    "DataSchema",
    "NonTemporal",
    "RelationSpec",
    "TableSpec",
    "table_clock",
]


@dataclass(frozen=True)
class NonTemporal:
    """Declares that a table has no time semantics at all.

    Written as the module-level singleton :data:`NON_TEMPORAL`. It is a *stated*
    absence, distinct from an unstated one: a table whose rows genuinely do not
    happen in time — a country reference table, a product catalogue — has no
    clock to declare, and saying so is different from forgetting to.

    A timestamp column can also be an ordinary attribute. Its type alone does
    not establish that it orders the rows of this table.
    """

    def __repr__(self) -> str:  # pragma: no cover - display only
        return "NON_TEMPORAL"


NON_TEMPORAL = NonTemporal()
"""The one legal way to say a table has no clock. See :class:`NonTemporal`."""


@dataclass(frozen=True)
class TableSpec:
    """One named table, how its rows are identified, and when they happened.

    ``time_column`` is **required**. A table names either the column that orders
    its rows, or :data:`NON_TEMPORAL` if its rows have no time semantics.
    Leaving it unstated is not expressible, because the two cases need opposite
    treatment and nothing else can distinguish them:

    - a temporal table whose clock was forgotten leaks the future into every
      window that reads it, silently;
    - a genuinely non-temporal table has no column to name, so demanding one
      would be unfollowable advice.

    ``columns`` declares every column available to the contract, including
    primary keys, relation keys, and the table clock. Physical sources may
    contain additional columns, but they are not part of this schema.

    Parquet column discovery requires the same column names in every file; an
    explicit mapping can instead select the shared columns. Selected types must
    agree in every file. ``InMemoryRows`` requires an explicit type for every
    declared column, even for empty or all-NULL data: types are never inferred
    from observed Python values. ``DataLoaderSpec.resolve`` replaces omitted
    types with concrete ones, so a saved contract records them.

    :param name: table name, as referenced by relations and specs.
    :param primary_key: column uniquely identifying each row; must be non-null
        and unique, and must appear in ``columns``.
    :param columns: selected columns and stored types. Omit to discover columns
        from Parquet metadata; use None for an individual inferred type.
    :param time_column: the column ordering this table's rows, or
        :data:`NON_TEMPORAL`. Keyword-only and required.
    """

    name: str
    primary_key: str
    columns: Mapping[str, DType | None] | None = None
    time_column: str | NonTemporal = dataclass_field(kw_only=True)

    def __post_init__(self) -> None:
        if not self.name or not self.primary_key:
            raise ValueError("table name and primary key must be non-empty")
        if isinstance(self.time_column, str):
            if not self.time_column:
                raise ValueError(
                    f"table {self.name!r} time column must be a column name or NON_TEMPORAL"
                )
        elif not isinstance(self.time_column, NonTemporal):
            # ``None`` is refused alongside every other value: an unstated clock
            # is the one thing this field exists to make inexpressible.
            raise TypeError(
                f"table {self.name!r} time column must be a column name or NON_TEMPORAL, "
                f"not {self.time_column!r}"
            )
        if self.columns is None:
            return
        if not isinstance(self.columns, Mapping):
            raise TypeError(f"table {self.name!r} columns must map names to DType declarations")
        columns = dict(self.columns)
        if not columns:
            raise ValueError(f"table {self.name!r} must declare its columns")
        if self.primary_key not in columns:
            raise ValueError(
                f"table {self.name!r} columns must include primary key {self.primary_key!r}"
            )
        if any(
            not name or dtype is not None and not isinstance(dtype, DType)
            for name, dtype in columns.items()
        ):
            raise ValueError(f"table {self.name!r} columns require names and DType declarations")
        if isinstance(self.time_column, str) and self.time_column not in columns:
            raise ValueError(
                f"table {self.name!r} time column {self.time_column!r} is not a declared column"
            )
        if (
            isinstance(self.time_column, str)
            and columns[self.time_column] is not None
            and not columns[self.time_column].temporal
        ):
            raise ValueError(
                f"table {self.name!r} clock requires timestamp or date32, "
                f"not {columns[self.time_column].name}"
            )
        object.__setattr__(self, "columns", _mapping(columns))


def table_clock(table: TableSpec) -> str | None:
    """The column a table is ordered by, or ``None`` when declared non-temporal.

    The single place that collapses the three-state declaration back to the
    two-state question most call sites actually ask.
    """

    return None if isinstance(table.time_column, NonTemporal) else table.time_column


@dataclass(frozen=True)
class RelationSpec:
    """Columns connecting two tables. A relation can be followed either way.

    A relation is an equality join of one column on each side; composite keys
    and other join predicates are not expressible. Neither key needs to be a
    primary key, but DiskGraph refuses a relation without exactly one
    primary-key side.

    :param name: name referenced by ``Rows.path``, ``GraphSample.relation`` and
        ``Field.path``.
    :param left_table: the side a graph traversal with ``direction="forward"``
        starts from; ``"reverse"`` starts from ``right_table``.
    """

    name: str
    left_table: str
    left_key: str
    right_table: str
    right_key: str


@dataclass(frozen=True)
class DataSchema:
    """The tables and relationships available to a sample recipe.

    Tables and relations are named definitions: their placement in the tuples
    has no meaning and does not change the contract's fingerprint. Every table
    names its clock or declares :data:`NON_TEMPORAL`.

    :param tables: every table the contract can read, each with a unique name.
    :param relations: equality joins between declared columns, each with a
        unique name.
    """

    tables: tuple[TableSpec, ...]
    relations: tuple[RelationSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", tuple(sorted(self.tables, key=lambda table: table.name)))
        object.__setattr__(
            self, "relations", tuple(sorted(self.relations, key=lambda relation: relation.name))
        )
        table_names = [table.name for table in self.tables]
        relation_names = [relation.name for relation in self.relations]
        if len(table_names) != len(set(table_names)):
            raise ValueError("table names must be unique")
        if len(relation_names) != len(set(relation_names)):
            raise ValueError("relation names must be unique")
        known_tables = set(table_names)
        by_name = {table.name: table for table in self.tables}
        for relation in self.relations:
            if relation.left_table not in known_tables or relation.right_table not in known_tables:
                raise ValueError(f"relation {relation.name!r} refers to an unknown table")
            for table_name, key in (
                (relation.left_table, relation.left_key),
                (relation.right_table, relation.right_key),
            ):
                if (
                    by_name[table_name].columns is not None
                    and key not in by_name[table_name].columns
                ):
                    raise ValueError(
                        f"relation {relation.name!r} key {table_name}.{key} is not declared"
                    )
