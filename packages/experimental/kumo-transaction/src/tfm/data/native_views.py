# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded physical source prefixes for exclusive and precomputed train splits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from tfm.contracts.errors import SchemaError
from tfm.contracts.table_source import TableSource
from tfm.data.native_source import EVENT_US

pc: Any = arrow_compute


@dataclass(frozen=True)
class SourceRows:
    """Select source rows once, independently of prediction-anchor multiplicity."""

    before_us: int | None = None
    partition_column: str | None = None
    partitions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.partition_column is None) != (not self.partitions):
            raise ValueError("Source partition selection requires a column and allowed values")


def source_view(
    source: TableSource, rules: Mapping[str, SourceRows], directory: Path
) -> TableSource:
    """Keep original normalized row IDs; never copy rows per prediction window."""
    if set(rules) - set(source.files):
        raise SchemaError("Native source view references an undeclared table")
    files = dict(source.files)
    directory.mkdir(parents=True, exist_ok=True)
    for ordinal, (name, rule) in enumerate(rules.items()):
        original = pq.ParquetFile(source.files[name])
        path = directory / f"table-{ordinal}.parquet"
        with pq.ParquetWriter(path, original.schema_arrow) as writer:
            for block in original.iter_batches(batch_size=4096):
                selected = pa.array([True] * block.num_rows)
                if rule.before_us is not None:
                    selected = pc.and_(selected, pc.less(block.column(EVENT_US), rule.before_us))
                if rule.partition_column is not None:
                    selected = pc.and_(
                        selected,
                        pc.is_in(
                            block.column(rule.partition_column), value_set=pa.array(rule.partitions)
                        ),
                    )
                writer.write_table(pa.Table.from_batches([block]).filter(selected))
        files[name] = path
    return replace(source, files=files)
