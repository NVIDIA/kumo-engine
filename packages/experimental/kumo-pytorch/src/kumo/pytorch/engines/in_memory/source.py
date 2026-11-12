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

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import groupby
from typing import Any

from kumo.pytorch.column_stats import Column, ColumnStats, StatRequest, data_sha256
from kumo.pytorch.contract.sample import BuiltSample, SampleRequest, SampleSpec
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.data.records import timestamp_to_epoch
from kumo.pytorch.encoding import TRAIN_BATCH
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
        import diskgraph
        import pyarrow as pa

        tables = {table.name: table for table in self.schema.tables}
        files = {}
        for (table, column), group in groupby(sorted(requests), key=lambda item: item[:2]):
            rows, clock = self._tables[table], table_clock(tables[table])
            stats = [stat for _, _, stat in group]
            dtype = tables[table].columns[column]
            computed = diskgraph.col_stats_from_arrow(
                # Microseconds, so sub-second instants round as ingest rounds them.
                pa.array(
                    [row.get(column) for row in rows],
                    type=pa.timestamp("us", "UTC")
                    if dtype.name == "timestamp"
                    else dtype.to_arrow(),
                ),
                None
                if clock is None
                else pa.array(
                    [
                        None if row.get(clock) is None else timestamp_to_epoch(row[clock])
                        for row in rows
                    ],
                    type=pa.int64(),
                ),
                cutoff,
                stats,
            )
            for stat in stats:
                files[(table, column, stat)] = {
                    "data": computed[stat],
                    "sha256": data_sha256(computed[stat]),
                }
        return ColumnStats(files, {"value": cutoff, "unit": "seconds", "inclusive": True})

    def column_text(
        self,
        columns: Sequence[Column],
        cutoff: int | None,
        name: str,
        train: Callable[[Iterator[Sequence[str]]], str],
    ) -> ColumnStats:
        """A tokenizer trained on string columns' stored values, as DiskGraph trains it."""

        tables = {table.name: table for table in self.schema.tables}

        def batches() -> Iterator[list[str]]:
            batch = []
            for table, column in columns:
                clock = table_clock(tables[table])
                for row in self._tables[table]:
                    if row.get(column) is not None and (
                        clock is None
                        or row.get(clock) is not None
                        and (cutoff is None or timestamp_to_epoch(row[clock]) <= cutoff)
                    ):
                        batch.append(row[column])
                        if len(batch) == TRAIN_BATCH:
                            yield batch
                            batch = []
            if batch:
                yield batch

        tokenizer = train(batches())
        sha256 = hashlib.sha256(tokenizer.encode()).hexdigest()
        return ColumnStats(
            {(*columns[0], name): {"tokenizer": tokenizer, "sha256": sha256}},
            {"value": cutoff, "unit": "seconds", "inclusive": True},
        )

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
