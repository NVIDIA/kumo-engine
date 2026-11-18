# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw samples and graph neighborhoods from rows held as Python mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kumo.pytorch.contract.sample import BuiltSample, NeighborhoodSpec, SampleRequest, SampleSpec
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.engine import GraphRoot
from kumo.pytorch.rows.clock import SECONDS, Clock
from kumo.pytorch.rows.graph import RowGraphSampler
from kumo.pytorch.rows.samples import RowSampleBuilder

__all__ = ["RowSource"]


class RowSource:
    """Builds samples and samples graphs from the given rows, with times in ``clock`` ticks."""

    def __init__(
        self,
        schema: DataSchema,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
        clock: Clock = SECONDS,
    ):
        self.schema = schema
        self._tables = tables
        self._clock = clock.unit
        self._cutoff = clock.cutoff
        self._samples = RowSampleBuilder(schema, tables, clock)
        self._graph = RowGraphSampler(schema, tables, clock)

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
        roots: Sequence[GraphRoot],
    ) -> Mapping[tuple[str, str], tuple[Any, ...]]:
        return self._graph.fitting_field_values(
            sample.entity.table,
            self._neighborhood(sample, spec),
            spec,
            roots,
        )

    def sample_graph(
        self,
        sample: SampleSpec,
        spec: Any,
        state: Any,
        roots: Sequence[GraphRoot],
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
    def _neighborhood(sample: SampleSpec, spec: Any) -> NeighborhoodSpec:
        neighborhood = sample.neighborhoods.get(spec.source)
        if neighborhood is None:
            raise ValueError(f"unknown neighborhood {spec.source!r}")
        return neighborhood
