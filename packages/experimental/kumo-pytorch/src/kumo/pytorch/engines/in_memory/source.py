# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-memory source: raw rows for one anchor, read from Python mappings.

Favors obvious scans and dictionaries over speed. It is the correctness
reference an optimized engine is compared against, which only works if it is
simple enough to be read and believed.

It sits behind the same `SampleSource` boundary every engine does, so the
comparison is a matter of swapping one object.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from kumo.pytorch.column_stats import (
    EMPTY_STATS,
    ColumnStats,
    StatRequest,
    TableColumnEncoder,
    file_name_part,
    read_column_stats,
)
from kumo.pytorch.contract.sample import BuiltSample, SampleRequest, SampleSpec
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.data.records import timestamp_to_epoch
from kumo.pytorch.encoding import TRAIN_BATCH, train_text
from kumo.pytorch.engines.in_memory.graph import InMemoryGraphSampler
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder

__all__ = ["InMemorySource"]


class InMemorySource:
    """Reads tables held as sequences of Python mappings."""

    def __init__(
        self,
        schema: DataSchema,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
    ):
        self.schema = schema
        self._tables = tables
        self._samples = InMemorySampleBuilder(schema, tables)
        self._graph = InMemoryGraphSampler(schema, tables)

    def build(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
    ) -> list[BuiltSample]:
        return self._samples.build(spec, requests)

    def fit_graph_fields(
        self,
        sample: SampleSpec,
        spec: Any,
        roots: Sequence[Any],
    ) -> Mapping[tuple[str, str], tuple[Any, ...]]:
        return self._graph.fitting_field_values(
            sample.entity.table,
            self._neighborhood(sample, spec),
            spec,
            roots,
        )

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
            None if cutoff is None else diskgraph.convert_ticks(cutoff, "nanoseconds", "seconds")
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
            cutoff = diskgraph.convert_ticks(cutoff, "nanoseconds", "seconds")

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

    def sample_graph(
        self,
        sample: SampleSpec,
        spec: Any,
        state: Any,
        roots: Sequence[Any],
        seed: int,
        epoch: int,
    ) -> Any:
        return self._graph.batch(
            sample.entity.table,
            self._neighborhood(sample, spec),
            spec,
            state,
            roots,
            seed=seed,
            epoch=epoch,
        )

    def close(self) -> None:
        """Nothing is held open; defined so every source closes the same way."""

    @staticmethod
    def _neighborhood(sample: SampleSpec, spec: Any):
        neighborhood = sample.neighborhoods.get(spec.source)
        if neighborhood is None:
            raise ValueError(f"unknown neighborhood {spec.source!r}")
        return neighborhood
