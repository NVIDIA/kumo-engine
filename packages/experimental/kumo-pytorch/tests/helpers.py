# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare package test fixtures from small row collections with either backend."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kumo.pytorch.anchors import ALL_ANCHORS
from kumo.pytorch.api import (
    AnchorTable,
    BatchSpec,
    DataLoaderSpec,
    Dataset,
    InMemoryEngine,
    InMemoryRows,
    Parquet,
    SampleRequest,
    SampleSpec,
    Task,
    prepare,
)
from kumo.pytorch.assembly import WRITTEN_BESIDE_STATE
from tests.native_fixtures import prepare_rows, request_anchor_table

__all__ = ["in_memory_dataset", "plan_diskgraph", "plan_in_memory"]


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


def parquet_anchors(path, columns, entity_id, as_of_time=None, **bindings):
    """Write a package test's anchor columns and bind them to the native reader."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(columns), path)
    return AnchorTable(Parquet(str(path)), entity_id, as_of_time, **bindings)


def in_memory_dataset(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dataset:
    return Dataset({name: InMemoryRows(rows) for name, rows in tables.items()})


def plan_in_memory(
    schema: Any,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    contract_or_sample: DataLoaderSpec | SampleSpec,
    requests_or_spec: Sequence[SampleRequest] | BatchSpec,
    requests: Sequence[SampleRequest] | None = None,
) -> Task:
    """Prepare and materialize a task over in-memory rows."""
    return Plans(schema, tables).prepare(contract_or_sample, requests_or_spec, requests)


def plan_diskgraph(
    schema: Any,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    contract_or_sample: DataLoaderSpec | SampleSpec,
    requests_or_spec: Sequence[SampleRequest] | BatchSpec,
    requests: Sequence[SampleRequest] | None = None,
) -> Task:
    """Prepare a DiskGraph task over the same rows using native Parquet anchors."""
    contract, fitting = _fixture_inputs(schema, contract_or_sample, requests_or_spec, requests)
    task = prepare_rows(contract, tables).materialize(
        request_anchor_table(fitting, schema=contract.schema), fit_split=_fit_everything(fitting)
    )
    assert callable(task._anchors.batches), "fixture must use native AnchorTable iteration"
    return task


def _fixture_inputs(schema, contract_or_sample, requests_or_spec, requests):
    if isinstance(contract_or_sample, DataLoaderSpec):
        return contract_or_sample, requests_or_spec
    return DataLoaderSpec(schema, contract_or_sample, requests_or_spec), requests


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
        contract, fitting = _fixture_inputs(
            self.schema, contract_or_sample, requests_or_spec, requests
        )
        return prepare(contract, self._dataset, InMemoryEngine()).materialize(
            fitting, fit_split=_fit_everything(fitting)
        )


class NativePlans:
    """Prepare the same row fixtures through Parquet and DiskGraph."""

    def __init__(self, schema, tables):
        self.schema = schema
        self.tables = tables

    def prepare(self, contract_or_sample, requests_or_spec, requests=None):
        return plan_diskgraph(
            self.schema, self.tables, contract_or_sample, requests_or_spec, requests
        )


def _fit_everything(requests: Sequence[SampleRequest]) -> Any:
    # These helpers fit on exactly the requests they are given; say so
    # explicitly when the requests carry splits, as materialize requires.
    return ALL_ANCHORS if any(request.split is not None for request in requests) else None


def written_state(artifact: Path) -> dict[str, Any]:
    """An artifact's ``fitted-state.json``, without what it keeps beside the state."""

    written = json.loads((Path(artifact) / "fitted-state.json").read_text())
    return {key: value for key, value in written.items() if key not in WRITTEN_BESIDE_STATE}
