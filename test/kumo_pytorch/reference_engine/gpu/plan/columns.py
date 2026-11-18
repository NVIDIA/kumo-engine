# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which tables and columns a contract reads, so that only those are loaded."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    from kumo.pytorch.spec import DataLoaderSpec


__all__ = ['needed_columns']


def needed_columns(contract: DataLoaderSpec) -> dict[str, set[str]]:
    """Each table the contract reads, and the columns it reads there, targets included.

    Taken from upstream Kumo PyTorch's own reading of the contract
    (``kumo.pytorch.serving_rows``), which lists every column a sample reads and
    why: primary keys, join keys, clocks and values.
    """

    from kumo.pytorch import serving_rows

    names = serving_rows._BatchNames(contract.sample, contract.batch)
    reads = [serving_rows._collect(contract, contract.sample, names)]
    reads.extend(found for _, found in serving_rows._target_reads(contract, names))
    needed: dict[str, set[str]] = defaultdict(set)
    for read in reads:
        for table, column in read.columns:
            needed[table].add(column)
    return dict(needed)


def _narrowed(contract: DataLoaderSpec) -> Any:
    """The contract's schema with only the tables and columns it reads (each table's primary key and clock kept)."""

    from kumo.pytorch.contract.schema import table_clock

    needed = needed_columns(contract)
    tables = []
    for table in contract.schema.tables:
        if table.name not in needed:
            continue
        keep = needed[table.name] | {table.primary_key} | ({table_clock(table)} - {None})
        tables.append(replace(table, columns={name: dtype for name, dtype in table.columns.items()
                                              if name in keep}))
    columns = {(table.name, column) for table in tables for column in table.columns}
    # a relation the contract follows has both keys among the columns read; the others are left out
    relations = tuple(relation for relation in contract.schema.relations
                      if {(relation.left_table, relation.left_key), (relation.right_table, relation.right_key)} <= columns)
    return replace(contract.schema, tables=tuple(tables), relations=relations)
