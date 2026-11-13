# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `InMemoryEngine` class."""

from __future__ import annotations

from collections.abc import Mapping

from kumo.pytorch.anchors import AnchorTable
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.dataset import Dataset, InMemoryRows, TableSource
from kumo.pytorch.engine import AnchorTask, SampleSource, register_engine
from kumo.pytorch.engines.in_memory.anchors import InMemoryAnchors
from kumo.pytorch.engines.in_memory.source import InMemorySource
from kumo.pytorch.spec import ValidationIssue

__all__ = ["InMemoryEngine"]


class InMemoryEngine:
    """Runs the contract in Python over `InMemoryRows`, for trying things out.

    Use it to play with the library on a few rows of data: print a batch or a
    contract to see what it looks like, or train a small model on a toy dataset.
    The tests use it too. It is not meant for real data or production use; use
    `DiskGraphEngine` for that. It evaluates the sampling rules with plain Python
    scans. Stored values keep their
    declared types, and declared primary keys must be unique and non-NULL.
    Equal-time graph neighbors at a fanout limit are ordered by the ``repr`` of
    their primary keys, or by source occurrence ordinal for a keyless table.
    """

    name = "in_memory"

    @staticmethod
    def check_sources(sources: Mapping[str, TableSource], path: str) -> tuple[ValidationIssue, ...]:
        """Refuse sources other than `InMemoryRows`, before any data is read."""
        unreadable = sorted({source.kind for source in sources.values()} - {"in_memory"})
        if not unreadable:
            return ()
        return (
            ValidationIssue(
                path,
                f"engine 'in_memory' does not read {unreadable} sources",
                "bind InMemoryRows, or use DiskGraphEngine for Parquet",
            ),
        )

    def open(self, dataset: Dataset, schema: DataSchema) -> SampleSource:
        tables = {}
        for table, source in dataset.tables.items():
            if not isinstance(source, InMemoryRows):
                raise TypeError(
                    f"table {table!r} is bound to a {source.kind!r} source; "
                    "this engine reads in-memory rows only"
                )
            tables[table] = source.rows
        return InMemorySource(schema, tables)

    def materialize_anchors(self, table: AnchorTable) -> AnchorTask:
        # The reference materializes lazily: rows keep their given order, which
        # becomes the canonical one, and labels are still computed per sample.
        return InMemoryAnchors(table)


register_engine(InMemoryEngine.name, InMemoryEngine)
