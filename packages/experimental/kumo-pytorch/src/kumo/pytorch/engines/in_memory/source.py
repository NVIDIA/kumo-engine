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

from collections.abc import Mapping, Sequence
from typing import Any

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
