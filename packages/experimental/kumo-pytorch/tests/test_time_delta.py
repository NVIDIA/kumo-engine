# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An event timestamp can become an ordinary numerical input field."""

from __future__ import annotations

import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api
from kumo.pytorch.api import (
    NON_TEMPORAL,
    BatchSpec,
    Columns,
    DataLoaderSpec,
    DataSchema,
    EntitySpec,
    EventSource,
    EventSpec,
    EventTimeDelta,
    Input,
    Padded,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    TableSpec,
    TimeWindow,
)
from kumo.pytorch.contract.batch import Encode, Fields
from kumo.pytorch.contract.dtypes import DType
from tests.helpers import Plans


def _contract(unit: str = "s") -> tuple[DataLoaderSpec, dict[str, list[dict[str, object]]]]:
    schema = DataSchema(
        (
            TableSpec("users", "user_id", {"user_id": DType("int64")}, time_column=NON_TEMPORAL),
            TableSpec(
                "events",
                "event_id",
                {
                    "event_id": DType("int64"),
                    "user_id": DType("int64"),
                    "at": DType("timestamp", unit=unit, timezone="UTC"),
                },
                time_column="at",
            ),
        ),
        (RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
    )
    sample = SampleSpec(
        EntitySpec("users", entity_type="user"),
        events={
            "history": EventSpec(
                Rows(("user_events",), TimeWindow("at")),
                {
                    "age_seconds": EventTimeDelta("seconds", dtype=DType("float64")),
                    "age_hours": EventTimeDelta("hours", dtype=DType("float64")),
                    "age_days": EventTimeDelta("days", dtype=DType("float64")),
                },
            )
        },
    )
    batch = BatchSpec(
        inputs={
            "history": Input(
                EventSource(("history",)),
                Fields(
                    fields=(
                        Encode(field="age_seconds", encoding=Raw()),
                        Encode(field="age_hours", encoding=Raw()),
                        Encode(field="age_days", encoding=Raw()),
                    ),
                    row=Columns(),
                ),
                Padded(),
            )
        }
    )
    return (
        DataLoaderSpec(schema, sample, batch),
        {
            "users": [{"user_id": 1}],
            "events": [
                {"event_id": 1, "user_id": 1, "at": 40},
                {"event_id": 2, "user_id": 1, "at": 90},
                {"event_id": 3, "user_id": 1, "at": 110},
            ],
        },
    )


def test_time_delta_is_anchor_minus_event_time_in_the_requested_unit() -> None:
    contract, tables = _contract()
    request = (SampleRequest("s1", 1, 100),)
    with Plans(contract.schema, tables).prepare(contract, request) as plan:
        batch = next(plan.batches(request, batch_size=1))
    history = batch.inputs["history"]
    assert history["age_seconds"]["values"].tolist() == [[60.0, 10.0]]
    assert history["age_hours"]["values"][0].tolist() == pytest.approx([60 / 3600, 10 / 3600])
    assert history["age_days"]["values"][0].tolist() == pytest.approx([60 / 86400, 10 / 86400])


@pytest.mark.parametrize("unit,ticks_per_second", [("us", 10**6), ("ns", 10**9)])
@pytest.mark.parametrize("days", [104, 105])
def test_native_event_age_across_long_histories(unit, ticks_per_second, days):
    contract, tables = _contract(unit)
    cutoff = 20_000_000
    elapsed = days * 86_400
    tables["events"] = [{"event_id": 1, "user_id": 1, "at": (cutoff - elapsed) * ticks_per_second}]
    with bound_tables(contract, tables, "diskgraph") as (dataset, engine):
        with api.prepare(contract, dataset, engine) as plan:
            with plan.materialize((SampleRequest("one", 1, cutoff),)) as task:
                batch = next(task.batches(batch_size=1, output_format="model"))
                age = batch.inputs["history"]["age_seconds"]
    assert age["valid"].tolist() == [[True]]
    assert age["values"].tolist() == [[elapsed]]


def test_time_delta_rejects_an_unknown_unit() -> None:
    with pytest.raises(ValueError, match="unit"):
        EventTimeDelta("fortnights", dtype=DType("float64"))


def test_time_delta_round_trips_as_part_of_the_contract() -> None:
    contract, _tables = _contract()
    assert DataLoaderSpec.from_json(contract.to_json()) == contract
