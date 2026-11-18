# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-owned Kumo engine using Polars for loader timing."""

from __future__ import annotations

from pathlib import Path

from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.dataset import Dataset

from ..base import EntityEngine, positive_threads
from ..source import EntitySource

__all__ = ["PolarsEngine"]


class PolarsEngine(EntityEngine):
    name = "nightly_polars"

    def __init__(
        self,
        *,
        num_threads: int | None = None,
        index: str | Path | None = None,
        ingest: str = "if_missing",
    ) -> None:
        self.num_threads = positive_threads(num_threads)
        self.configure(index=index, ingest=ingest)

    def open(self, dataset: Dataset, schema: DataSchema) -> EntitySource:
        self.validate_binding(dataset, schema)
        schema = dataset.resolve_schema(schema)
        from ..artifacts import open_prepared
        from .store import PolarsStore
        from .temporal import TemporalQueries

        store = open_prepared(
            PolarsStore,
            dataset,
            schema,
            backend="polars",
            index=self.index,
            ingest=self.ingest,
            num_threads=self.num_threads,
        )
        try:
            return EntitySource(store, schema, TemporalQueries(store, schema))
        except BaseException:
            store.close()
            raise
