# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for running a contract in memory.

Tests care about what a contract produces, not about binding storage, so this
collapses `prepare` and `materialize` into one call over the requests that
vary. It is a convenience for tests only — the public path is
`prepare(contract, dataset, engine).materialize(anchors)`. The plan is left
open for the task's lifetime; in memory there is nothing to release.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kumo.pytorch.anchors import ALL_ANCHORS
from kumo.pytorch.api import (
    BatchSpec,
    DataLoaderSpec,
    Dataset,
    InMemoryEngine,
    InMemoryRows,
    SampleRequest,
    SampleSpec,
    Task,
    prepare,
)
from kumo.pytorch.assembly import WRITTEN_BESIDE_STATE

__all__ = ["in_memory_dataset", "plan_in_memory"]


def reorder_dicts(value: Any, *, shuffled: bool = False) -> Any:
    """Vary JSON object placement without changing any array positions."""

    rng = random.Random(0)

    def reorder(item):
        if isinstance(item, dict):
            pairs = list(item.items())
            if shuffled:
                rng.shuffle(pairs)
            else:
                pairs.reverse()
            return {key: reorder(child) for key, child in pairs}
        if isinstance(item, list):
            return [reorder(child) for child in item]
        return item

    return reorder(value)


def in_memory_dataset(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dataset:
    return Dataset({name: InMemoryRows(rows) for name, rows in tables.items()})


def plan_in_memory(
    schema: Any,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    contract_or_sample: DataLoaderSpec | SampleSpec,
    requests_or_spec: Sequence[SampleRequest] | BatchSpec,
    requests: Sequence[SampleRequest] | None = None,
) -> Task:
    """Prepare and materialize a task over in-memory rows.

    Accepts either a complete contract, or a sample and batch spec that are
    assembled into one, so tests that predate `DataLoaderSpec` read the same.
    """

    if isinstance(contract_or_sample, DataLoaderSpec):
        contract = contract_or_sample
        fitting = requests_or_spec
    else:
        contract = DataLoaderSpec(schema, contract_or_sample, requests_or_spec)
        fitting = requests
    return prepare(contract, in_memory_dataset(tables), InMemoryEngine()).materialize(
        fitting, fit_split=_fit_everything(fitting)
    )


class Plans:
    """Bind in-memory rows once, prepare many plans from them."""

    def __init__(self, schema: Any, tables: Mapping[str, Sequence[Mapping[str, Any]]]):
        self.schema = schema
        self._dataset = in_memory_dataset(tables)

    def prepare(
        self,
        contract_or_sample: DataLoaderSpec | SampleSpec,
        requests_or_spec: Sequence[SampleRequest] | BatchSpec,
        requests: Sequence[SampleRequest] | None = None,
    ) -> Task:
        if isinstance(contract_or_sample, DataLoaderSpec):
            contract, fitting = contract_or_sample, requests_or_spec
        else:
            contract = DataLoaderSpec(self.schema, contract_or_sample, requests_or_spec)
            fitting = requests
        return prepare(contract, self._dataset, InMemoryEngine()).materialize(
            fitting, fit_split=_fit_everything(fitting)
        )


def _fit_everything(requests: Sequence[SampleRequest]) -> Any:
    # These helpers fit on exactly the requests they are given; say so
    # explicitly when the requests carry splits, as materialize requires.
    return ALL_ANCHORS if any(request.split is not None for request in requests) else None


def written_state(artifact: Path) -> dict[str, Any]:
    """An artifact's ``fitted-state.json``, without what it keeps beside the state."""

    written = json.loads((Path(artifact) / "fitted-state.json").read_text())
    return {key: value for key, value in written.items() if key not in WRITTEN_BESIDE_STATE}
