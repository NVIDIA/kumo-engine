# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview exposes produced raw values and the exact model-facing batch."""

from __future__ import annotations

import json
from dataclasses import replace
from inspect import signature
from typing import Any

import pytest
import torch

from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    Batching,
    BatchSpec,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    EventSource,
    EventSpec,
    Field,
    Fields,
    Graph,
    GraphLayer,
    GraphSample,
    InMemoryEngine,
    InMemoryRows,
    Input,
    NeighborhoodSource,
    NeighborhoodSpec,
    Padded,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TimeWindow,
    ValidationError,
    Vocabulary,
    prepare,
    preview,
)
from kumo.pytorch.engines.in_memory.source import InMemorySource


def _fixture() -> tuple[DataLoaderSpec, dict[str, list[dict[str, Any]]]]:
    schema = DataSchema(
        (
            TableSpec(
                "users",
                "user_id",
                {"user_id": DType("int64"), "segment": DType("string")},
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "events",
                "event_id",
                {
                    "event_id": DType("int64"),
                    "user_id": DType("int64"),
                    "at": DType("timestamp", unit="s", timezone="UTC"),
                    "kind": DType("string"),
                },
                time_column="at",
            ),
        ),
        (RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
    )
    sample = SampleSpec(
        EntitySpec("users", entity_type="user"),
        entity_values={
            "segment": Field(column="segment"),
            "past_count": Aggregate(
                Rows(("user_events",), TimeWindow("at")), "count", dtype=DType("int64")
            ),
        },
        events={
            "history": EventSpec(
                Rows(("user_events",), TimeWindow("at")), {"kind": Field(column="kind")}, limit=2
            )
        },
        neighborhoods={
            "activity": NeighborhoodSpec(
                (GraphLayer((GraphSample("user_events", fanout=2, strategy="recent"),)),)
            )
        },
        targets={
            "future_count": Aggregate(
                Rows(("user_events",), TimeWindow("at", 0, 50)), "count", dtype=DType("int64")
            )
        },
    )
    batch = BatchSpec(
        inputs={
            "entity": Input(
                EntitySource(),
                Fields(
                    fields=(
                        Encode(field="segment", encoding=Vocabulary()),
                        Encode(field="past_count", encoding=Raw()),
                    )
                ),
                Separate(),
            ),
            "history": Input(
                EventSource(("history",)),
                Fields(fields=(Encode(field="kind", encoding=Vocabulary()),)),
                Padded(),
            ),
            "graph": Input(NeighborhoodSource("activity"), Graph(), None),
        }
    )
    tables = {
        "users": [{"user_id": 1, "segment": "gold"}],
        "events": [
            {"event_id": 1, "user_id": 1, "at": 60, "kind": "login"},
            {"event_id": 2, "user_id": 1, "at": 90, "kind": "purchase"},
            {"event_id": 3, "user_id": 1, "at": 110, "kind": "future"},
        ],
    }
    return (DataLoaderSpec(schema, sample, batch), tables)


def _equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    else:
        assert left == right


def _dataset(tables: dict[str, list[dict[str, Any]]]) -> Dataset:
    return Dataset({name: InMemoryRows(rows) for name, rows in tables.items()})


def test_prepared_preview_returns_the_exact_normal_batch_without_mutating_state() -> None:
    contract, tables = _fixture()
    request = (SampleRequest("s1", 1, 100),)
    plan = prepare(contract, _dataset(tables), InMemoryEngine()).materialize(request)
    before = plan.manifest["fitted_state_fingerprint"]
    result = preview(plan, request, seed=7)
    normal = next(plan.batches(request, batch_size=1, seed=7))
    _equal(result.batch.inputs, normal.inputs)
    _equal(result.batch.targets, normal.targets)
    assert plan.manifest["fitted_state_fingerprint"] == before
    assert result.encoding_summary["entity"]["summary"]["segment"]["entries"] == [["gold", 2]]
    plan.close()


def test_preview_shows_only_produced_raw_values_and_the_encoded_batch() -> None:
    contract, tables = _fixture()
    request = (SampleRequest("s1", 1, 100),)
    with prepare(contract, _dataset(tables), InMemoryEngine()).materialize(request) as plan:
        result = preview(plan, request)
    payload = result.to_dict()
    sample = payload["samples"][0]
    assert sample["entity"]["values"] == {"segment": "gold", "past_count": 2.0}
    assert sample["events"] == {
        "history": [
            {"timestamp": 60, "values": {"kind": "login"}},
            {"timestamp": 90, "values": {"kind": "purchase"}},
        ]
    }
    assert sample["targets"] == {"future_count": 1.0}
    assert "selections" not in sample
    assert "event_ordering" not in sample
    assert payload["batch"]["inputs"]["graph"]["nodes"]["events"]["ids"]["shape"] == [2]
    assert "inputs: entity, history, graph" in str(result)


def test_contract_preview_temporarily_fits_encodings_and_returns_one_batch() -> None:
    contract, tables = _fixture()
    request = (SampleRequest("s1", 1, 100),)
    dataset = _dataset(tables)
    result = preview(contract, dataset, InMemoryEngine(), request)
    with prepare(contract, dataset, InMemoryEngine()).materialize(request) as expected:
        normal = next(expected.batches(request, batch_size=1))
    _equal(result.batch.inputs, normal.inputs)
    _equal(result.batch.targets, normal.targets)
    assert result.encoding_summary["history"]["summary"]["kind"]["entries"] == [
        ["login", 2],
        ["purchase", 3],
    ]
    json.dumps(result.to_dict())


def test_contract_preview_rejects_a_dataset_for_a_different_schema() -> None:
    contract, _ = _fixture()
    dataset = Dataset({"other": InMemoryRows(({"id": 1},))})
    with pytest.raises(ValidationError, match="dataset"):
        preview(contract, dataset, InMemoryEngine(), (SampleRequest("s1", 1, 100),))


def test_preview_has_a_non_configurable_eight_request_limit() -> None:
    contract, tables = _fixture()
    accepted = tuple(SampleRequest(f"s{index}", 1, 100) for index in range(8))
    rejected = tuple(SampleRequest(f"s{index}", 1, 100) for index in range(9))
    dataset = _dataset(tables)
    result = preview(contract, dataset, InMemoryEngine(), accepted)
    assert len(result.samples) == 8
    assert result.batch.metadata["entity_ids"].shape == (8,)
    assert "max_samples" not in signature(preview).parameters
    with pytest.raises(ValueError, match="limited to 8"):
        preview(contract, dataset, InMemoryEngine(), rejected)


def test_preview_puts_every_request_in_one_batch_regardless_of_normal_policy() -> None:
    contract, tables = _fixture()
    contract = replace(contract, batch=replace(contract.batch, batching=Batching(max_samples=1)))
    requests = tuple(SampleRequest(f"s{index}", 1, 100) for index in range(3))
    plan = prepare(contract, _dataset(tables), InMemoryEngine()).materialize(requests)
    assert len(list(plan.batches(requests))) == 3
    result = preview(plan, requests)
    assert len(result.samples) == 3
    assert result.batch.metadata["entity_ids"].shape == (3,)
    plan.close()


def test_contract_preview_closes_its_temporary_source(monkeypatch: pytest.MonkeyPatch) -> None:
    contract, tables = _fixture()
    closed: list[InMemorySource] = []
    original_close = InMemorySource.close

    def track_close(source: InMemorySource) -> None:
        closed.append(source)
        original_close(source)

    monkeypatch.setattr(InMemorySource, "close", track_close)
    preview(contract, _dataset(tables), InMemoryEngine(), (SampleRequest("s1", 1, 100),))
    assert len(closed) == 1
