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

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kumo.pytorch.column_stats import EMPTY_STATS, ColumnStats, StatRequest, stype_groups
from kumo.pytorch.contract.sample import BuiltSample, SampleRequest, SampleSpec
from kumo.pytorch.contract.schema import DataSchema
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

        import pyarrow as pa
        import pyarrow.parquet as pq

        from kumo.pytorch.dataset import Dataset, Parquet
        from kumo.pytorch.materialization import ensure_diskgraph_index

        columns = {}
        with tempfile.TemporaryDirectory() as directory:
            for table in self.schema.tables:
                wanted = [item for item in requests if item[0] == table.name]
                rows = self._tables[table.name]
                if not wanted or not rows:
                    columns.update({item: EMPTY_STATS for item in wanted})
                    continue
                # Microseconds, so sub-second instants round as ingest rounds them.
                types = {
                    name: pa.timestamp("us", "UTC")
                    if dtype.name == "timestamp"
                    else dtype.to_arrow()
                    for name, dtype in table.columns.items()
                }
                path = f"{directory}/{table.name}.parquet"
                pq.write_table(pa.Table.from_pylist(rows, pa.schema(types)), path)
                schema = DataSchema(tables=(table,))
                for number, group in enumerate(stype_groups(wanted)):
                    index = Path(directory, f"{table.name}-{number}")
                    ensure_diskgraph_index(
                        schema,
                        Dataset({table.name: Parquet(path)}),
                        index,
                        column_stats=(group, [cutoff]),
                    )
                    (stats,) = index.glob("col_stats/*/col_stats.json")
                    found = json.loads(stats.read_text())[table.name]
                    columns.update({item: found[item[1]] for item in group})
        return ColumnStats(columns, {"value": cutoff, "unit": "seconds"})

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
