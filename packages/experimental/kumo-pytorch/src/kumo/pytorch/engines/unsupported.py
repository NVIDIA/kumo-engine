# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the DiskGraph engine doesn't support yet, declared in one list.

Each entry is refused with its code and next step, at the stage it names:
`check()` refuses it before any data is opened (and so does `prepare()`);
``"prepare()"`` entries are refused while `prepare` binds the files;
``"batches()"`` entries are refused when batches are requested, before any batch
is delivered. docs/limitations.md lists the entries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

__all__ = ["DISKGRAPH_UNSUPPORTED", "Unsupported"]


@dataclass(frozen=True)
class Unsupported:
    """One thing an engine refuses today, and what to do instead.

    :param code: stable issue code, carried by every refusal as `ValidationIssue.code`.
    :param description: what is refused, in plain words.
    :param next_step: what to do instead.
    :param refused_at: when it is refused: ``"check()"`` (by `check` and `prepare`,
        before any data is opened), ``"prepare()"`` (while `prepare` binds the
        files) or ``"batches()"`` (when batches are requested, before any batch).
    """

    code: str
    description: str
    next_step: str
    refused_at: Literal["check()", "prepare()", "batches()"] = "check()"

    def error(self, detail: str) -> ValueError:
        """The refusal raised outside `check`, naming the code and the next step."""

        return ValueError(f"{detail} [{self.code}]; {self.next_step}")


_ENTRIES = (
    Unsupported(
        "DG-SOURCE-KIND",
        "Tables and anchor tables must be bound to Parquet files; other sources, "
        "such as `InMemoryRows`, are refused.",
        "write the rows to a local Parquet file and bind it with `Parquet(path)`",
    ),
    Unsupported(
        "DG-REMOTE-PARQUET",
        "The DiskGraph engine doesn't support remote Parquet paths, such as `s3://`, "
        "yet, for tables or anchor tables.",
        "copy the files to a local disk and bind the local path",
    ),
    Unsupported(
        "DG-RELATION-KEYS",
        "A relation must have exactly one side that is its own table's primary key; "
        "joins of two foreign keys, and of two primary keys, are refused.",
        "for two foreign keys, relate each to a key table whose primary key is the "
        "shared key; for two primary keys, add a copy of the key as an ordinary "
        "column of one table and relate through that column",
    ),
    Unsupported(
        "DG-TIMED-INTERMEDIATE",
        "A `Rows` path can't pass through an intermediate table that has a clock, "
        "because the cutoff can't be applied at an intermediate hop.",
        "use a direct relation if it gives the same rows, or precompute the joined "
        "rows into one table before writing the Parquet files",
    ),
    Unsupported(
        "DG-ROW-FILTER",
        "Row filters, in `Rows.filters` or `EntitySpec.filters`, are refused.",
        "precompute the condition as a column, or write only the matching rows to "
        "their own table, before writing the Parquet files",
    ),
    Unsupported(
        "DG-ACTIVE-INTERVAL",
        "Active intervals (`Rows.active`) are refused.",
        "write one event row per change of state, with its time as the table clock",
    ),
    Unsupported(
        "DG-ENTITY-VALUE-KIND",
        "The DiskGraph engine doesn't support entity values that look up a field "
        "through a relation path or by its own time column yet.",
        "precompute the looked-up value as a column of the root table",
    ),
    Unsupported(
        "DG-JOINED-FIELD",
        "Event and row-target fields must read the event row itself; a field with a "
        "relation path or its own time column is refused.",
        "precompute the joined column onto the event table before writing the Parquet files",
    ),
    Unsupported(
        "DG-DATED-ENTITY-VALUES",
        "The DiskGraph engine doesn't support `SampleSpec.dated_entity_values` yet.",
        "declare an event input (`EventSpec`) over the same `Rows` instead",
    ),
    Unsupported(
        "DG-CURRENT-EVENTS",
        "The DiskGraph engine doesn't support `SampleSpec.current_events` yet.",
        "read the event through an ordinary historical event input",
    ),
    Unsupported(
        "DG-CANDIDATE-SETS",
        "Candidate sets, in `SampleSpec.candidate_sets` or as candidate columns of "
        "an anchor table, are refused.",
        "build the candidates' features with a second contract whose entity is the "
        "candidate table, anchored at the same times, and pair them in your own code",
    ),
    Unsupported(
        "DG-UNIFORM-GRAPH",
        "Uniform graph sampling isn't supported yet: DiskGraph's native uniform mode "
        "doesn't yet keep each example to rows at or before its own prediction time.",
        'use `strategy="recent"`',
    ),
    Unsupported(
        "DG-GRAPH-CLOCK-OVERRIDE",
        "The DiskGraph engine doesn't support a graph traversal `time_column` other "
        "than the destination table's clock yet.",
        "use the destination table's declared clock",
    ),
    Unsupported(
        "DG-STATIC-TIMED-TRAVERSAL",
        'A graph traversal into a table with a clock can\'t be `temporal="static"`.',
        'use `temporal="inherit"`',
    ),
    Unsupported(
        "DG-REDUCTION-OPERAND",
        "Reductions over decimal or bytes columns are refused; direct decimal field "
        "reads are supported.",
        "cast decimals to float64 or integer units, and bytes to a string or integer "
        "key, before writing the Parquet files",
    ),
    Unsupported(
        "DG-COLUMN-RENAME",
        "`Parquet(columns=...)` renames in a table binding are refused; physical "
        "column names must match the declared ones. Anchor tables accept renames.",
        "rename the columns in the Parquet files to the declared names",
        refused_at="prepare()",
    ),
    Unsupported(
        "DG-INT64-CLOCK",
        "A column declared as a timestamp must be stored as a Parquet timestamp; an "
        "int64 column of epoch values is refused.",
        "write the column as a Parquet timestamp in the declared unit",
        refused_at="prepare()",
    ),
    Unsupported(
        "DG-NESTED-COLUMN",
        "Parquet columns of list, struct or map type are refused; every declared "
        "column must be stored as a scalar.",
        "flatten the values into scalar columns, or into rows of their own table, "
        "before writing the Parquet files",
        refused_at="prepare()",
    ),
    Unsupported(
        "DG-DUPLICATE-LAYER-RELATION",
        "A graph layer can't sample the same relation twice, including two relations "
        "that resolve to the same stored relation.",
        "sample the relation once in the layer, with the combined fanout",
        refused_at="batches()",
    ),
)

#: The DiskGraph engine's unsupported list, keyed by code, in declaration order.
DISKGRAPH_UNSUPPORTED: Mapping[str, Unsupported] = MappingProxyType(
    {entry.code: entry for entry in _ENTRIES}
)
