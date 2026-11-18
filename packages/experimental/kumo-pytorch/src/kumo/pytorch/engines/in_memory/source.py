# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-memory source: raw rows for one anchor, read from Python mappings.

Favors obvious scans and dictionaries over speed, so it stays simple enough to
be read and believed. It sits behind the same `SampleSource` boundary every
engine does, and adds the column statistics and text fitting reads to the row
assembly serving shares.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from kumo.pytorch.column_stats import (
    EMPTY_STATS,
    ColumnStats,
    StatRequest,
    TableColumnEncoder,
    file_name_part,
    read_column_stats,
)
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.data.records import timestamp_to_epoch
from kumo.pytorch.encoding import TRAIN_BATCH, train_text
from kumo.pytorch.rows.source import RowSource

__all__ = ["InMemorySource"]


class InMemorySource(RowSource):
    """Reads tables held as sequences of Python mappings, on a whole-second clock."""

    def column_stats(self, requests: Sequence[StatRequest], cutoff: int | None) -> ColumnStats:
        import tempfile

        import diskgraph
        import pyarrow as pa
        import pyarrow.parquet as pq

        from kumo.pytorch.dataset import Dataset, Parquet
        from kumo.pytorch.materialization import ensure_diskgraph_index

        stored = [
            table
            for table in self.schema.tables
            if self._tables[table.name] and any(item[0] == table.name for item in requests)
        ]
        columns = {item: EMPTY_STATS for item in requests}
        ticks = (
            None
            if cutoff is None
            else diskgraph.convert_ticks(cutoff, "nanoseconds", "seconds", rounding="floor")
        )
        if not stored:
            return ColumnStats(columns, {"value": ticks, "unit": "seconds"})
        with tempfile.TemporaryDirectory() as directory:
            sources = {}
            for table in stored:
                rows = [
                    {name: dtype.normalize(row.get(name)) for name, dtype in table.columns.items()}
                    for row in self._tables[table.name]
                ]
                types = pa.schema(
                    [(name, dtype.to_arrow()) for name, dtype in table.columns.items()]
                )
                path = f"{directory}/{file_name_part(table.name)}.parquet"
                pq.write_table(pa.Table.from_pylist(rows, types), path)
                sources[table.name] = Parquet(path)
            index = Path(directory, "index")
            ensure_diskgraph_index(DataSchema(tables=tuple(stored)), Dataset(sources), index)
            names = {table.name for table in stored}
            group = [item for item in requests if item[0] in names]
            columns.update(read_column_stats(index, group, ticks, "seconds").columns)
        return ColumnStats(columns, {"value": ticks, "unit": "seconds"})

    def column_text(self, encoder: TableColumnEncoder, cutoff: int | None) -> tuple[str, None]:
        import diskgraph

        tables = {table.name: table for table in self.schema.tables}
        if cutoff is not None:
            cutoff = diskgraph.convert_ticks(cutoff, "nanoseconds", "seconds", rounding="floor")

        def batches() -> Iterator[list[str]]:
            batch = []
            for table, column in encoder.columns:
                clock = table_clock(tables[table])
                for row in self._tables[table]:
                    if row.get(column) is not None and (
                        clock is None
                        or row.get(clock) is not None
                        and (cutoff is None or timestamp_to_epoch(row[clock]) <= cutoff)
                    ):
                        batch.append(encoder.dtype.normalize(row[column], path=f"{table}.{column}"))
                        if len(batch) == TRAIN_BATCH:
                            yield batch
                            batch = []
            if batch:
                yield batch

        return train_text(encoder.config, batches()), None
